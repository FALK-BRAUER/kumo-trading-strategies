"""Every runner can halt itself on a daily loss, and says so when it cannot (kumo-trading-platform issue 548, #111).

TWO DEFECTS, ONE RULE.

**#548 — TECHIVOL-005 could not halt itself on risk under any condition.** `pgrunner` enforced
`RiskLimits.daily_loss_frac`; `qc27_runner` read one of that dataclass's ten fields and it was not
this one. `momentum_rotation.broker_equity` calls this limit "the only automatic stop this strategy
has", so the gap was a live question, not a tidiness one.

The worse half is that it left NO TRACE. "this lane had no reason to halt" and "this lane has no
ability to halt" produce the same journal — nothing — so no operator reading it could tell them
apart. A control that fails by never firing does not fail loudly; it fails by continuing.

**#111 — the lanes that DID have the stop had it silently disarmed by a NaN.** Measured through the
real `PgSessionRunner.run` before the fix:

    good baseline, 99% fall  ->  halted
    NaN  baseline, 99% fall  ->  NOT halted, blocked later by an unrelated ranking floor

EXPOSURE WAS NIL — do not read what follows as a caught incident. Cockpit measured both production
databases afterwards: 55 decision rows, 55 finite anchors, zero non-finite ever; and the production
write path cannot represent the poison at all (a float NaN is a bare token jsonb rejects, a Decimal
raises inside `json.dumps`, and `PgJournal.write` turns either into a refused session). The
demonstration below ran through a FIXTURE journal that accepts anything — a double more permissive
than production, which is the failure this file's own siblings exist to prevent.

What survives is the CLASS. A STRINGIFIED "Infinity" IS valid JSON, does persist, and would have
made `now < inf * 0.95` always true — a permanent spurious halt, a worse failure than the disarmed
one. It needs one caller with a `default=str` encoder. These tests guard non-finite anchors by any
route, not one bug.

`_finite_equity` guarded the equity read at decision time. `_session_start_equity` guarded nothing:
`(r.get("detail") or {}).get("equity")` sees a NaN as truthy, returns `float(NaN)` as the baseline,
and then `now < NaN * 0.95` is False like every comparison with a NaN. `Decimal("NaN")` off Alpaca
is not exotic — cockpit measured one on 2026-08-28 — and the poisoned anchor persists into the NEXT
session's check too.

WHY ONE FILE. Both are the same root: two derivations of "an equity worth comparing against a
limit". `daily_loss` is now the only one.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from kumo_strategies.runtime.executor import daily_loss
from kumo_strategies.runtime.executor.daily_loss import Action, Baseline, decide, finite

from kumo_strategies.strategies import _layout


# ==================================================================================================
# THE RULE — pure, so no fixture can be what makes it pass
# ==================================================================================================

@pytest.mark.parametrize("value,expected", [
    (100_000.0, 100_000.0),
    (0.0, 0.0),                       # a REAL account value, not an absence
    (float("nan"), None),
    (float("inf"), None),
    (None, None),
    ("", None),
])
def test_finite_is_the_only_definition_of_a_usable_equity(value, expected):
    assert finite(value) == expected


@pytest.mark.parametrize("start,now,frac,expected", [
    (100_000.0, 50_000.0, 0.05, Action.HALT),
    (100_000.0, 99_000.0, 0.05, Action.PROCEED),
    (100_000.0, 95_000.0, 0.05, Action.PROCEED),   # EXACTLY at the threshold is not through it
    (100_000.0, None,     0.05, Action.BLOCK),     # unverifiable is not within the limit
    (None,      50_000.0, 0.05, Action.PROCEED),   # a lane's first session has no anchor
    (100_000.0, 50_000.0, None, Action.PROCEED),   # unarmed
])
def test_the_decision_table(start, now, frac, expected):
    assert decide(start, now, frac).action is expected


def test_a_breach_says_BOTH_numbers():
    """"HALTED" alone sends an operator to read the source at the worst possible moment."""
    why = decide(100_000.0, 50_000.0, 0.05).reason
    assert "50.0%" in why and "5%" in why


# ==================================================================================================
# THE ANCHOR — missing and corrupt are OPPOSITE facts
# ==================================================================================================

class _Jrn:
    def __init__(self, rows):
        self.rows = rows

    async def tail(self, n=400, kind=None, symbol=None):
        return self.rows


def _baseline(rows):
    return asyncio.run(daily_loss.baseline(_Jrn(rows), "2026-08-28"))


def test_this_sessions_anchor_wins():
    assert _baseline([{"session": "2026-08-28", "detail": {"equity": 100_000.0}}]) \
        == (100_000.0, Baseline.FOUND)


def test_the_previous_sessions_anchor_is_CARRIED():
    """Without this the limit cannot fire at all on a once-a-day strategy: the baseline is written by
    the first decision of the session, and the first decision is the only one. An overnight gap from
    100k to 93k passed straight through, and 93k was then recorded as today's start, making the loss
    invisible on every later comparison too."""
    assert _baseline([{"session": "2026-08-27", "detail": {"equity": 99_000.0}}]) \
        == (99_000.0, Baseline.CARRIED)


def test_NO_ROWS_is_MISSING_and_must_stay_permissive():
    """A lane's first session ever has no prior decision. Blocking it would mean a new lane can never
    start, so this one absence — and only this one — reads as permission."""
    assert _baseline([]) == (None, Baseline.MISSING)


def test_a_RECORDED_NaN_is_POISONED_not_missing():
    """The two are told apart — and BOTH proceed, which took a measurement to get right.

    My first version blocked on this. It deadlocks: a BLOCK returns before the DECISION write, so
    the session records no new anchor, and the next session reads the same poisoned rows. Measured
    over three consecutive sessions: block, block, block, with nothing an operator could do about
    it. `test_a_poisoned_journal_does_not_DEADLOCK` below is that measurement, kept.

    They stay distinct because the operator needs different words: "this lane has never written an
    anchor" is a new lane, "every anchor it wrote is garbage" is a broken account feed.
    """
    assert _baseline([{"session": "2026-08-27", "detail": {"equity": float("nan")}}]) \
        == (None, Baseline.POISONED)
    assert _baseline([]) == (None, Baseline.MISSING)


def test_a_row_with_NO_equity_key_is_MISSING_not_poisoned():
    """A DECISION row that carries no anchor is not a broken anchor. After the write-side guard
    (`anchor()` omits the key when non-finite) this is what a poisoned write looks like going
    forward, and it must not be reported as a corrupt feed forever."""
    assert _baseline([{"session": "2026-08-27", "detail": {"hold": ["AAA"]}}]) \
        == (None, Baseline.MISSING)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_the_WRITE_side_omits_what_the_READ_side_would_reject(value):
    """One invariant, both directions: **the `equity` key is present if and only if it is usable.**

    #111 existed because the writer recorded whatever the broker returned and the reader trusted the
    key's presence. Omitting beats writing null: the reader's scan already treats an absent key as
    "not a record", so legacy rows and new rows need one rule, not two.
    """
    assert daily_loss.anchor(value) == {}
    assert daily_loss.anchor(100_000.0) == {"equity": 100_000.0}


def test_a_GOOD_prior_anchor_beats_a_corrupt_current_one():
    """A NaN today does not throw away a usable anchor from yesterday. The old code did the reverse —
    it returned the current session's NaN and disarmed the limit."""
    assert _baseline([{"session": "2026-08-28", "detail": {"equity": float("nan")}},
                      {"session": "2026-08-27", "detail": {"equity": 99_000.0}}]) \
        == (99_000.0, Baseline.CARRIED)


# ==================================================================================================
# THE CLASS — every runner, discovered
# ==================================================================================================

def _has_machinery(path: Path) -> bool:
    """Can this runner hold the property at all? Asked STRUCTURALLY, never by name.

    The stop needs three things: a `journal` to read its anchor from and write its refusal to, a
    `lifecycle` to halt, and `limits` carrying the threshold. `TemplateSessionRunner` has none of
    them — it is a decision-and-sizing reference, not a session gateway — so demanding a halt of it
    would be demanding a row for an event it cannot have.

    A name-based skip (`!= "template_runner.py"`) would say the same thing today and go on saying it
    after the template gained a journal and a lifecycle. This sweeps it back in automatically on the
    day it can hold the property, which is the only version of the exclusion that cannot rot.
    """
    tree = ast.parse(path.read_text())
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)
                  and isinstance(n.target, ast.Name)}
        if {"journal", "lifecycle", "limits"} <= fields:
            return True
    return False


def _runners() -> list[Path]:
    out = sorted(p for p in _layout.runner_files() if _has_machinery(p))   # ks#211
    assert len(out) >= 2, f"discovery found only {out}; every assertion below would be vacuous"
    return out


def _method_calls(path: Path) -> set[str]:
    """Every `<...>.<name>(...)` this module calls, from the parse tree.

    A helper rather than an inline walk so the assertion compares against a set of AST attribute
    names, not against a local bound from `read_text()` — `test_no_new_test_asserts_a_substring_
    against_source` tracks the ORIGIN of a value, and correctly cannot tell the two apart inside one
    function.
    """
    return {n.func.attr for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


def _reads_limits(path: Path) -> set[str]:
    return {n.attr for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Attribute) and n.value.attr == "limits"}


@pytest.mark.parametrize("path", _runners(), ids=_layout.rel)
def test_every_runner_reaches_the_shared_stop(path):
    """Aimed at the class. The ticket named QC345 and TECHIVOL; the fix has to survive the third
    runner nobody has written yet, which is the difference between fixing the reported lane and
    fixing the class."""
    assert "enforce" in _method_calls(path), (
        f"{_layout.rel(path)} never calls `daily_loss.enforce`, so the lane it runs cannot halt itself on "
        f"risk under any condition, and leaves no journal row saying so (kumo-trading-platform issue 548).")


@pytest.mark.parametrize("path", _runners(), ids=_layout.rel)
def test_every_runner_WRITES_the_anchor_its_own_stop_reads(path):
    """THE WIRING TEST, and the one this fix would have shipped broken without.

    `daily_loss.baseline` reads `detail["equity"]` off the lane's own DECISION rows. qc27's decision
    basis had no such key — so wiring the stop to it would have produced `start=None` on every
    session forever and a halt that could never fire, while every test above stayed green. A
    mechanism wired to a producer that feeds it nothing is the built-never-executed shape this stop
    was added to end, re-created by the fix for it.

    Bound to the AST of the dict actually handed to the DECISION write, not to the file containing
    the word "equity" — this module's own docstrings say it a dozen times.

    This is the CHEAP cross-runner sweep. The expensive one that cannot be fooled by either
    representation is `test_the_anchor_written_in_one_session_HALTS_the_next` in
    test_qc27_runner.py, which drives two real sessions and injects nothing.
    """
    tree = ast.parse(path.read_text())
    anchored = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("write", "_j"):
            continue
        if not any(isinstance(a, ast.Name) and a.id == "DECISION" for a in node.args):
            continue
        for kw in node.keywords:
            if kw.arg != "detail":
                continue
            # the detail dict may be built inline or bound to a name; resolve a bare name once
            target = kw.value
            if isinstance(target, ast.Name):
                name = target.id          # captured BEFORE the rebind below, or the next
                for n in ast.walk(tree):  # iteration reads `.id` off an ast.Dict
                    if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                            and any(isinstance(t, ast.Name) and t.id == name
                                    for t in n.targets)):
                        target = n.value
            if isinstance(target, ast.Dict):
                # EITHER a literal key OR the shared write-side helper unpacked into the dict
                # (`**daily_loss.anchor(...)`, which omits the key when the value is unusable).
                # Binding to the literal alone would fail the moment the runners were routed
                # through the guard that makes the key trustworthy — a test that punishes the fix.
                anchored |= any(isinstance(k, ast.Constant) and k.value == "equity"
                                for k in target.keys)
                anchored |= any(k is None and isinstance(v, ast.Call)
                                and getattr(v.func, "attr", None) == "anchor"
                                for k, v in zip(target.keys, target.values))
    assert anchored, (
        f"{_layout.rel(path)} writes a DECISION row without an `equity` key, so `daily_loss.baseline` finds "
        f"no anchor on any session and the daily-loss stop can never fire. It would look wired and "
        f"be dead.")


def test_the_stop_is_not_reimplemented_anywhere():
    """One derivation. #111 exists because there were two, and only one of them guarded a NaN."""
    # SCOPED TO LOSS HALTS. A blanket ban on `.halt(` would go red on the first legitimate
    # disconnect halt — `lifecycle.py` documents HALTED as the lost-feed state too — and a guard
    # that fires on correct code is a guard someone deletes. Bound to what the halt is ABOUT.
    offenders = []
    for path in _runners():
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "halt"):
                continue
            words = " ".join(v.value for a in node.args for v in ast.walk(a)
                             if isinstance(v, ast.Constant) and isinstance(v.value, str))
            if "loss" in words.lower():
                offenders.append(f"{_layout.rel(path)}: halt({words[:40]}...)")
    assert not offenders, (
        f"{offenders} call `lifecycle.halt` directly instead of going through `daily_loss.enforce`. "
        f"A second copy of this rule is how the guarded and unguarded halves of 'a usable equity' "
        f"diverged in the first place.")


def test_the_TEMPLATE_is_excluded_for_a_REASON_that_expires():
    """VACUITY GUARD on the exclusion above, and a record of a real gap.

    `template_runner.py` is what a new lane is copied from, and it has no journal, no lifecycle and
    no limits — so a lane copied from it starts with no risk machinery of any kind. That is how
    TECHIVOL-005 arrived where it did.

    This asserts the exclusion is structural, not a name: if the template ever gains the three
    fields, it joins the sweep and must then carry the stop like everything else. Filed separately;
    giving the template a lifecycle is a design change, not part of this fix.
    """
    template = _layout.strategies_root() / "template" / "runner.py"
    assert template.exists(), "the template moved; this guard is now checking nothing"
    assert not _has_machinery(template), (
        "template_runner now has journal/lifecycle/limits, so it CAN halt and must. Delete this "
        "test — the sweep above already covers it.")


# ==================================================================================================
# THINGS THE EXTRACTION QUIETLY CHANGED — each one measured, then fixed, then pinned here
# ==================================================================================================

def test_a_BROKER_THAT_RAISES_refuses_the_session_instead_of_crashing_it():
    """`_finite_equity` wrapped `float(self.broker.equity())`, so a broker whose `equity()` raised
    became a BLOCK. The extraction evaluated the callable OUTSIDE the guard and passed the result
    in, which let that exception escape the session path — a refusal turned into a crash, the one
    direction this module must never move. Measured escaping before the fix.

    There were TWO unguarded calls: the second was in the BLOCK message itself, which quotes the raw
    value, so the refusal path re-invoked the broker that was refusing.
    """
    said = []

    class J:
        async def tail(self, n=400, kind=None, symbol=None):
            return [{"session": "2026-08-27", "detail": {"equity": 100_000.0}}]

    async def w(kind, summary, detail=None):
        said.append(summary)

    def boom():
        raise TypeError("no account snapshot yet")

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    v = asyncio.run(daily_loss.enforce(journal=J(), write=w, lifecycle=Lifecycle(State.TRADING, "t"),
                                       equity=boom, frac=0.05, session="2026-08-28"))
    assert v.action is Action.BLOCK
    assert any("CANNOT BE CHECKED" in s for s in said)
    assert any("TypeError" in s for s in said), (
        "the refusal does not say what the broker actually did, so an operator cannot tell a "
        "missing snapshot from a broken adapter")


def test_ZERO_is_a_real_equity_but_not_a_usable_ANCHOR():
    """`finite(0.0)` is 0.0 and that is RIGHT for `now` — zero is a real account value. As a
    baseline it is vacuous-but-armed: `now < 0 * 0.95` is never true, so HALT becomes unreachable
    while the status row says "anchored on 0". Measured: a total loss against a 0.0 anchor
    PROCEEDED.

    The old truthy scan skipped 0.0 rows and fell through to an older anchor. Keeping that is the
    only reading where the stop still works.
    """
    assert finite(0.0) == 0.0, "zero must remain a real equity on the `now` side"
    assert _baseline([{"session": "2026-08-27", "detail": {"equity": 0.0}}]) \
        == (None, Baseline.POISONED)


def test_a_zero_anchor_does_not_SHADOW_an_older_usable_one():
    """The scan continues past it, exactly as it does past a NaN — one unusable row must never hide
    a finite anchor sitting behind it."""
    assert _baseline([{"session": "2026-08-27", "detail": {"equity": 0.0}},
                      {"session": "2026-08-26", "detail": {"equity": 100_000.0}}]) \
        == (100_000.0, Baseline.CARRIED)


def test_a_RAISING_persistence_hook_still_leaves_the_HALT_row():
    """The local halt already happened, which is the fail-safe direction. A raising hook must not
    also cost us the row saying the lane halted — that leaves it stopped with no record of why."""
    said = []

    class J:
        async def tail(self, n=400, kind=None, symbol=None):
            return [{"session": "2026-08-27", "detail": {"equity": 100_000.0}}]

    async def w(kind, summary, detail=None):
        said.append(summary)

    async def bad_hook(reason):
        raise RuntimeError("cockpit unreachable")

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    halted = []

    class Lc(Lifecycle):
        def halt(self, why):
            halted.append(why)

    v = asyncio.run(daily_loss.enforce(journal=J(), write=w, lifecycle=Lc(State.TRADING, "t"),
                                       equity=10_000.0, frac=0.05, session="2026-08-28",
                                       on_halt=bad_hook))
    assert v.action is Action.HALT and halted, "the local halt did not happen"
    assert any("HALTED:" in s for s in said), "the halt row was lost to the failing hook"
    assert any("could not be PERSISTED" in s for s in said), (
        "nothing said the halt is session-local, so an operator would believe it is durable")


@pytest.mark.parametrize("returned,persisted", [
    (None, True), (True, True),
    ("OPERATOR_WON", False), ("CONTRADICTION", False), (False, False),
])
def test_a_persistence_hook_that_RETURNS_a_failure_is_not_treated_as_success(returned, persisted):
    """Cockpit's `save_if_unchanged` reports OPERATOR_WON / CONTRADICTION by RETURNING them, not by
    raising. Reacting only to exceptions would let an UNPERSISTED halt read as persisted — absence
    readable as permission, in the persistence path of a risk stop.

    Found by cockpit reviewing what I merged; `enforce` inspected only for raises.
    """
    said = []

    class J:
        async def tail(self, n=400, kind=None, symbol=None):
            return [{"session": "2026-08-27", "detail": {"equity": 100_000.0}}]

    async def w(kind, summary, detail=None):
        said.append(summary)

    async def hook(reason):
        return returned

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    asyncio.run(daily_loss.enforce(journal=J(), write=w, lifecycle=Lifecycle(State.TRADING, "t"),
                                   equity=10_000.0, frac=0.05, session="2026-08-28", on_halt=hook))
    complained = any("NOT PERSISTED" in s for s in said)
    assert complained is not persisted, (
        f"a hook returning {returned!r} was reported as "
        f"{'persisted' if not complained else 'unpersisted'} — wrong either way round")
    assert any("HALTED:" in s for s in said), "the halt row went missing"
