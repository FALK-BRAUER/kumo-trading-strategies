"""Tests for `QC27SessionRunner` — the gateway that had never executed (#33, #63).

WHY THIS FILE EXISTS AT ALL. `qc27_runner.py` was written, reviewed and left untracked with zero
tests. That is the exact defect class that produced QC345's `KeyError: 'eligible'` on its first live
session and MOMENTUM's 22 fills with zero terminal rows — code that exists, reads correctly, and has
never run. Four architecture reviews on this codebase found 0 of 6 such bugs. So every guard below
is asserted against a double that can actually FAIL, and each was mutation-bitten: the guard was
broken, the test watched go red, and the guard restored.

THE ORDER OF OPERATIONS IS THE SAFETY MODEL. These tests pin the ORDER, not just the outcomes —
idempotency before lifecycle, journal before broker, re-read lifecycle before submitting, exits
before entries. An implementation that does all the same things in a different order is a different
safety model, and most of these tests would still pass on outcomes alone.
"""

from __future__ import annotations

import asyncio
import sys

import pandas as pd
import pytest

from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
from kumo_strategies.runtime.executor.pgjournal import (
    DECISION, ERROR, ORDER, RISK, DuplicateDecision)
from kumo_strategies.strategies.qc27_tech_inverse_vol.runner import DECISION_SLOT, QC27SessionRunner
from kumo_strategies.runtime.executor.runner import RiskLimits
from kumo_strategies.strategies.qc27_tech_inverse_vol import QC27TechInverseVolConfig

SESSION = "2025-06-02"
LIVE_CFG = QC27TechInverseVolConfig(momentum_price_field="close")


# -- doubles that can fail ------------------------------------------------------------------------
class Jrn:
    """Records the ORDER of writes, because the order is the thing under test. `write` returns a
    truthy row id like the real one: returning None means "not journalled", which the runner treats
    as fail-closed, so a double that returned None would make every send look correctly refused."""

    def __init__(self, *, decided=False, raise_on_decision=False, fail_intent=False):
        self.rows: list[tuple[str, str, str | None]] = []
        # SLOT IS RECORDED, because production stores it and a double that drops a column cannot
        # observe that column being wrong. This one dropped it, which is precisely why the runner
        # could accept a `slot=` argument and file every row under a module constant instead, for
        # five days, while every test here passed.
        self.slots: list[str | None] = []
        self.asked_slots: list[str | None] = []
        #: (kind, session, detail) for every row, so `tail` can answer the way
        #: production does rather than returning a fixed list.
        self.decisions: list[tuple[str, str, dict]] = []
        self._decided, self._raise = decided, raise_on_decision
        self._fail_intent = fail_intent

    async def tail(self, n=400, kind=None, symbol=None):
        """PRODUCTION HAS THIS, so the double must. `daily_loss.baseline` reads the lane's own
        DECISION rows for the equity anchor, and a double missing the method raised AttributeError
        out of the session path the moment the daily-loss stop was wired in.

        Returns the rows this double has actually recorded, not a convenient constant — a fake more
        forgiving than production is how a live defect stays invisible. Empty until a DECISION with
        an equity anchor has been written, which is exactly production's answer on a lane's first
        session.
        """
        return [{"session": ses, "detail": det}
                for k, ses, det in self.decisions if kind is None or k == kind]

    # `correlation` MATCHES `PgJournal.write`, which has always accepted it. This double was
    # NARROWER than production and so failed a change production would have taken — the
    # loud direction, unlike a permissive double, but still a double that is not the thing.
    async def write(self, kind, summary, *, session, detail=None, symbol=None, slot=None,
                    correlation=None):
        if kind == DECISION and self._raise:
            raise DuplicateDecision(f"{session}/{slot} already claimed")
        if self._fail_intent and detail and detail.get("phase") == "intent":
            return None
        self.rows.append((kind, summary, symbol))
        self.decisions.append((kind, session, detail or {}))
        self.slots.append(slot)
        return len(self.rows)

    async def decided_this_session(self, session, slot=None):
        self.asked_slots.append(slot)
        return self._decided

    def kinds(self) -> list[str]:
        return [k for k, _, _ in self.rows]


class Brk:
    def __init__(self, *, positions=None, equity=100_000.0, ok=True, detail="filled"):
        self._pos = dict(positions or {})
        self._eq, self._ok, self._detail = equity, ok, detail
        self.sent: list[tuple[str, str, int]] = []

    def submit(self, req):
        self.sent.append((req.side, req.symbol, req.qty))
        # OrderResult is (ok, order_id, DETAIL, request) -- `.detail`, never `.reason`. Reading the
        # field that does not exist made every refusal read "refused" in production, discarding the
        # one thing that says why.
        return type("R", (), {"ok": self._ok, "order_id": "x", "detail": self._detail})()

    async def exit(self, req):
        return self.submit(req)

    def positions(self):
        """The ACCOUNT's book. Present on the double because it is present in production — a double
        that omitted it would have hidden the ownership defect rather than exposed it."""
        return dict(self._pos)

    def strategy_positions(self):
        """What THIS strategy owns. Defaults to the whole of `_pos`, i.e. "everything on this double
        is ours", which is the right baseline for tests that are not about ownership. The ownership
        tests override it to make the two differ, which is the only configuration that can catch a
        gateway reading the wrong one."""
        return dict(self._pos)

    def equity(self):
        return self._eq

    def last_price(self, symbol, **kw):
        """`last_price`, matching the REAL broker.

        This double implemented `price`, which exists on no broker — so the runner's call to
        `self.broker.price(...)` worked here and raised AttributeError in production, where a bare
        `except Exception: return 0` turned it into "0 shares". Every entry TECHIVOL-005 formed sized
        to zero, and 26 tests were green because the double had been built to match the CALLER rather
        than the callee. A double that agrees with the code cannot test the code."""
        return 100.0


def _panel(n_symbols: int = 12, n_sessions: int = 140) -> pd.DataFrame:
    """Enough history to clear the 100-session warmup AND the 63-session lookback. A shorter panel
    makes the runner return "no feature rows" and every downstream assertion passes vacuously."""
    dates = pd.bdate_range("2025-01-01", periods=n_sessions)
    rows = []
    for i in range(n_symbols):
        base = 50.0 + i
        for j, d in enumerate(dates):
            px = base * (1.0 + 0.004 * j + 0.01 * ((i + j) % 5) / 5)
            rows.append({"ticker": f"SYM{i:02d}", "date": d, "open": px, "high": px * 1.01,
                         "low": px * 0.99, "close": px, "volume": 5_000_000.0})
    return pd.DataFrame(rows)


PANEL = _panel()
REB = pd.Timestamp(PANEL["date"].iloc[-1])


def _runner(*, state=State.TRADING, jrn=None, brk=None, **kw) -> tuple[QC27SessionRunner, Jrn, Brk]:
    j, b = jrn or Jrn(), brk or Brk()
    r = QC27SessionRunner(journal=j, lifecycle=Lifecycle(state, "test"), cfg=LIVE_CFG, broker=b,
                          limits=kw.pop("limits", RiskLimits()), **kw)
    return r, j, b


def _run(r, panel=None, session=None):
    # The runner picks its own cadence off the config, so the session under test must BE a rebalance
    # date for that cadence rather than an arbitrary day.
    from kumo_strategies.strategies.qc27_tech_inverse_vol import rebalance_dates
    p = PANEL if panel is None else panel
    s = session or str(sorted(rebalance_dates(p["date"], LIVE_CFG.rebalance_period))[-1].date())
    return asyncio.run(r.run(p, s))


# -- 1. idempotency comes FIRST --------------------------------------------------------------------
def test_an_already_decided_session_does_not_decide_again():
    """A restart mid-session must not decide the same slot twice. First, before lifecycle is even
    read — otherwise a HALT would mask a double-decide rather than the other way round."""
    r, j, b = _runner(jrn=Jrn(decided=True))
    res = _run(r)
    assert res.decided is False
    assert DECISION_SLOT in res.blocked and "already decided" in res.blocked
    assert b.sent == [], "submitted on a session that had already decided"
    assert DECISION not in j.kinds()


def test_idempotency_is_checked_before_the_lifecycle():
    """Order matters: both a decided session and a HALTED state block, so outcomes alone cannot tell
    them apart. The BLOCKED REASON is what distinguishes them, and it must name the slot."""
    r, _, _ = _runner(state=State.HALTED, jrn=Jrn(decided=True))
    assert "already decided" in _run(r).blocked, "lifecycle was read before idempotency"


# -- 2. lifecycle ----------------------------------------------------------------------------------
@pytest.mark.parametrize("state", [State.DISABLED, State.WARMUP, State.HALTED])
def test_a_non_deciding_state_neither_decides_nor_submits(state):
    r, j, b = _runner(state=state)
    res = _run(r)
    assert res.decided is False and b.sent == []
    assert RISK in j.kinds(), "a refusal that leaves no journal row is invisible to the operator"


def test_shadow_decides_and_publishes_but_submits_nothing():
    """SHADOW is the dry-run state and the whole point of it is that the decision path RUNS. A
    SHADOW that skips the decision proves nothing about the strategy before it trades."""
    r, j, b = _runner(state=State.SHADOW)
    res = _run(r)
    assert res.decided is True
    assert DECISION in j.kinds()
    assert b.sent == [], "SHADOW submitted an order"


def test_liquidating_exits_but_never_enters():
    """The wind-down state. Exits must still run — a liquidation that could not sell is the worst of
    both — while entries are refused."""
    r, j, b = _runner(state=State.LIQUIDATING, brk=Brk(positions={"SYM00": 10, "SYM01": 5}))
    _run(r)
    assert all(side == "SELL" for side, _, _ in b.sent), f"LIQUIDATING entered: {b.sent}"


# -- 3. cadence comes from the CONFIG, not a hardcoded month ---------------------------------------
def test_a_non_rebalance_session_is_blocked_and_is_not_a_failure():
    r, _, b = _runner()
    res = _run(r, session="2025-06-11")
    assert res.decided is False and "not a rebalance session" in res.blocked
    assert b.sent == []


def test_the_cadence_is_read_from_the_config_not_hardwired_monthly():
    """THE DEFECT THIS PINS. Both production call sites -- the adapter and this gateway -- called
    `rebalance_dates()` with no period and were hardwired monthly, while the backtest took a
    `rebalance_period` argument. A cadence sweep could report weekly and the live strategy would
    still rebalance monthly, with nothing failing to say so.

    Weekly rebalance dates that are NOT month starts are exactly the sessions a hardwired-monthly
    gateway refuses, so this fails if the config stops being read."""
    from kumo_strategies.strategies.qc27_tech_inverse_vol import rebalance_dates
    weekly_cfg = QC27TechInverseVolConfig(momentum_price_field="close", rebalance_period="W")
    monthly = set(rebalance_dates(PANEL["date"], "M"))
    weekly_only = [d for d in rebalance_dates(PANEL["date"], "W") if d not in monthly]
    assert weekly_only, "fixture cannot distinguish the two cadences"

    r = QC27SessionRunner(journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "t"), cfg=weekly_cfg,
                          broker=Brk())
    res = asyncio.run(r.run(PANEL, str(weekly_only[-1].date())))
    assert res.decided is True, "weekly config still refusing a weekly rebalance — cadence ignored"


def test_the_cadence_reaches_the_artifact_tag():
    """Two runs differing only in cadence would otherwise write to one artifact path and silently
    overwrite each other."""
    tags = {QC27TechInverseVolConfig(rebalance_period=p).tag() for p in ("M", "W", "D")}
    assert len(tags) == 3, "cadence does not reach the tag; artifacts collide"


# -- 4. the decision is journalled BEFORE any order ------------------------------------------------
def test_the_decision_is_journalled_before_the_broker_is_touched():
    """"Why did it buy X" must be answerable months later without re-running anything — and it must
    be answerable even if the process dies between deciding and submitting."""
    r, j, b = _runner(brk=Brk(positions={"SYM00": 10}))
    _run(r)
    assert DECISION in j.kinds()
    assert j.kinds().index(DECISION) < j.kinds().index(ORDER), "order journalled before decision"


def test_a_concurrent_claim_on_the_slot_stops_rather_than_submits():
    """DuplicateDecision means another run already claimed this slot. Two runners submitting off one
    slot builds the book twice."""
    r, _, b = _runner(jrn=Jrn(raise_on_decision=True))
    res = _run(r)
    assert res.decided is False and b.sent == []


def test_the_journalled_basis_carries_what_the_decision_was_made_on():
    r, _, _ = _runner()
    res = _run(r)
    for key in ("hold", "enter", "exit", "weights", "scores", "eligible_name_days"):
        assert key in res.detail, f"basis is missing {key!r}"


# -- 5. last look before money moves ---------------------------------------------------------------
def test_a_lifecycle_that_moved_while_the_session_ran_blocks_submission():
    """The state read at step 2 can be seconds old. An operator who HALTs in that window must not
    watch it submit anyway."""
    async def moved():
        return State.HALTED
    r, j, b = _runner(recheck_state=moved, brk=Brk(positions={"SYM00": 10}))
    res = _run(r)
    assert res.decided is True, "the decision itself should still stand"
    assert b.sent == [], "submitted under a state the operator had left"
    assert RISK in j.kinds()


def test_a_recheck_that_raises_fails_CLOSED():
    """Cannot confirm the operator's intent -> do not act on a stale copy of it. The tempting
    implementation swallows the error and proceeds on the state it already had."""
    async def boom():
        raise RuntimeError("db gone")
    r, j, b = _runner(recheck_state=boom, brk=Brk(positions={"SYM00": 10}))
    _run(r)
    assert b.sent == [], "submitted despite being unable to confirm the lifecycle"
    assert RISK in j.kinds()


# -- 6. submission ---------------------------------------------------------------------------------
def test_exits_are_submitted_before_entries():
    """Selling first frees cash AND the venue's share reservation before anything asks for them, so
    a rotation cannot be half-applied into a shortfall."""
    # Holding a name the strategy does not want forces an exit, while holding none of the target
    # forces entries -- so this session has BOTH. The obvious fixture (hold all twelve) yields two
    # exits and zero entries, which made the ordering assertion vacuous: a mutation deleting every
    # exit still passed, because the guard `if "SELL" in sides and "BUY" in sides` simply skipped.
    # Caught by mutation bite, not by review.
    r, _, b = _runner(brk=Brk(positions={"ZZZ": 10}))
    _run(r)
    sides = [s for s, _, _ in b.sent]
    assert "SELL" in sides, "fixture produced no exit — the ordering assertion would be vacuous"
    assert "BUY" in sides, "fixture produced no entry — the ordering assertion would be vacuous"
    assert max(i for i, s in enumerate(sides) if s == "SELL") < \
           min(i for i, s in enumerate(sides) if s == "BUY"), f"entered before exiting: {sides}"


def test_sizing_uses_the_strategys_allocation_not_the_account():
    """A 20k strategy on a 100k account must size off 20k. Sizing off the account is how it builds
    two oversized positions where the research measured ten."""
    r, _, b = _runner(brk=Brk(equity=100_000.0), limits=RiskLimits(allocated_equity=20_000.0))
    _run(r)
    buys = [q for s, _, q in b.sent if s == "BUY"]
    assert buys, "no entries to size"
    # price is 100.0, so notional = qty * 100 and the whole book must fit inside the allocation.
    assert sum(buys) * 100.0 <= 20_000.0 * 1.01, \
        f"sized off account equity, not the 20k allocation: {sum(buys)*100.0:,.0f}"


def test_a_ZERO_allocation_does_not_size_off_the_whole_account():
    """`allocated_equity=0.0` means ALLOCATED NOTHING, and 0.0 is falsy.

    `equity = self.limits.allocated_equity or self._account_equity()` treated it as "unset" and fell
    through to the account's book, so a lane granted nothing sized every entry off the FULL account
    — the exact failure the module contract forbids: "Sizing off the whole account is how a 20k
    strategy on a 100k account builds two oversized positions where the research measured ten."

    Live exposure when this was found (2026-08-24, reported by kumo-trading-platform): on kumo-staging,
    BCTROT-004 is funded at 100000 and EVERY OTHER ALLOCATION IS 0. Any of those lanes deciding
    there would not have sized to zero — it would have sized to the whole account, against
    BCTROT's capital. It had not bitten only because `exec_action_log` on that stack is empty and
    no lane has ever decided.

    `None` (never configured) keeps falling through to the account, unchanged: that is what every
    running config passes, and `test_sizing_falls_back_to_the_account_when_no_allocation_is_set`
    holds it. Zero is not absence.

    Degrading to inaction is already the runner's own behaviour once equity is 0 — `notional <= 0`
    skips the symbol — so no new refusal path is needed, only stopping the fallback from firing.
    """
    r, _, b = _runner(brk=Brk(equity=100_000.0), limits=RiskLimits(allocated_equity=0.0))
    _run(r)
    buys = [(sym, q) for s, sym, q in b.sent if s == "BUY"]
    assert buys == [], (
        f"a lane allocated 0.0 sized off the 100k account and would have submitted {buys} — "
        f"{sum(q for _, q in buys) * 100.0:,.0f} of someone else's capital")


def test_sizing_falls_back_to_the_account_when_no_allocation_is_set():
    """The other half, and the reason zero must be handled separately rather than by deleting the
    fallback: `None` means cockpit configured no allocation, and falling through to the account is
    today's behaviour for every running config. If this goes red the zero fix has moved a live book.
    """
    r, _, b = _runner(brk=Brk(equity=100_000.0), limits=RiskLimits())
    _run(r)
    assert [q for s, _, q in b.sent if s == "BUY"], \
        "an unset allocation stopped entering — the fallback was deleted, not narrowed"


def test_a_NON_FINITE_account_equity_sizes_NOTHING_rather_than_raising_mid_submit():
    """`float(self.broker.equity() or 0.0)` has the same trap as everywhere else today: **NaN is
    truthy**, so the `or` never fires and a nan comes straight back.

    A nan equity then makes `notional = equity * weight` nan, and the `notional <= 0` skip is False
    for a nan — so it reaches `_qty`, where `int(nan / px)` raises `ValueError: cannot convert float
    NaN to integer` PART WAY THROUGH the entry loop, after earlier symbols have already been sent.
    A half-submitted rotation is worse than none.

    The nautilus adapters now return None for a non-finite equity, so this is defence in depth rather
    than the only guard — but `Broker` is a protocol and nothing stops another implementation
    returning a nan directly.
    """
    r, _, b = _runner(brk=Brk(equity=float("nan")), limits=RiskLimits())
    _run(r)
    assert [s for s, _, _ in b.sent if s == "BUY"] == [], \
        "sized entries against a non-finite account equity"


def test_the_budget_gate_can_refuse_an_entry_and_the_refusal_says_why():
    async def gate(symbol, notional):
        return False, "over strategy target"
    r, _, b = _runner(budget_gate=gate)
    res = _run(r)
    assert [s for s, _, _ in b.sent if s == "BUY"] == []
    assert "over strategy target" in (res.blocked or ""), "refusal reason was discarded"


def test_an_unjournalled_intent_is_never_sent():
    """FAIL CLOSED. If the intent write fails we have not traded — and `resume`, which decides what
    was attempted from those records, would otherwise send it again. No record, no order."""
    r, _, b = _runner(jrn=Jrn(fail_intent=True), brk=Brk(positions={"SYM00": 10}))
    _run(r)
    assert b.sent == [], "sent an order whose intent was never journalled"


def test_a_refusal_reports_the_venues_detail_not_the_word_refused():
    """`OrderResult` is (ok, order_id, DETAIL, request). Reading `.reason` — which does not exist —
    made every refusal read "refused", discarding "not a subscribed instrument" and the like."""
    r, j, _ = _runner(brk=Brk(ok=False, detail="not a subscribed instrument"))
    res = _run(r)
    assert "not a subscribed instrument" in (res.blocked or "") or \
           any("not a subscribed instrument" in s for _, s, _ in j.rows)


# -- 7. the terminal row -------------------------------------------------------------------------
def test_record_terminal_writes_the_venues_actual_answer():
    """The `phase=result` row is submit-time only — Nautilus accepting the order, not the venue
    filling it. MOMENTUM's caller was wired for weeks while no gateway defined this method, so every
    call no-opped through a getattr: 22 fills, zero terminal rows (kumo-trading-platform issue 383)."""
    r, j, _ = _runner()
    asyncio.run(r.record_terminal(SESSION, "SYM00", True, "filled 10 @ 100"))
    assert any(d for k, s, sym in j.rows for d in [s] if "terminal" in s and sym == "SYM00")


def test_a_rejected_terminal_is_journalled_as_an_error_not_an_order():
    r, j, _ = _runner()
    asyncio.run(r.record_terminal(SESSION, "SYM00", False, "rejected: wash trade"))
    assert (ERROR, "terminal: SYM00 rejected — rejected: wash trade", "SYM00") in j.rows


# -- 8. the decision slot names the time it actually fires -----------------------------------------
def test_the_decision_slot_names_the_configured_fill_time():
    """It was the literal "open+5m" while the shipped offset is 150 minutes. The slot is the
    idempotency key AND the string cockpit parses to schedule the alert (`qc345.py`'s
    `_open_offset_from_settings` reads this exact `open+Nm` shape), so a stale literal schedules the
    decision at the wrong time and labels the journal with an hour it never traded at."""
    from kumo_strategies.strategies.qc27_tech_inverse_vol import OPEN_OFFSET_MINUTES
    assert DECISION_SLOT == f"open+{OPEN_OFFSET_MINUTES}m"
    assert DECISION_SLOT.startswith("open+") and DECISION_SLOT.endswith("m")
    assert int(DECISION_SLOT.removeprefix("open+").removesuffix("m")) == OPEN_OFFSET_MINUTES


# -- 9. the lifecycle must be read per SESSION, not captured at construction ------------------------
def test_a_supplied_reader_beats_the_constructed_lifecycle():
    """`lifecycle` is captured when the runner is built — once, at node boot. Live that would freeze
    the operator's state for the lifetime of the process, so a HALT written on Tuesday would not be
    seen on Wednesday. Constructed TRADING here, read HALTED: the read must win."""
    async def halted():
        return State.HALTED
    r, j, b = _runner(state=State.TRADING, read_state=halted)
    res = _run(r)
    assert res.decided is False and b.sent == []
    assert "HALTED" in (res.blocked or "")


def test_the_reader_is_consulted_every_session_not_cached():
    """An implementation that reads once and remembers passes the test above and still freezes."""
    # NOTE each session reads TWICE — the opening read and the last look — so the double cannot be
    # a two-element iterator. That it is called four times across two sessions is itself the
    # evidence that both looks happen and neither is cached.
    seen = []

    async def changing():
        seen.append(len(seen))
        return State.TRADING
    r, _, _ = _runner(state=State.TRADING, read_state=changing)
    _run(r)
    first = len(seen)
    _run(r)
    assert first >= 1, "state was never read"
    assert len(seen) > first, "state was read once and cached across sessions"


def test_wiring_only_read_state_still_gets_the_last_look():
    """THE TRAP. `recheck_state` is a separate field, so a caller who wires the per-session read and
    forgets it would get fresh state at the top and then submit without ever looking again — the
    last look silently absent on the only path with real money behind it. It must fall back."""
    calls = []

    async def reader():
        # TRADING for the opening read, HALTED by the time the last look happens.
        calls.append(1)
        return State.TRADING if len(calls) == 1 else State.HALTED
    r, _, b = _runner(state=State.TRADING, read_state=reader,
                      brk=Brk(positions={"ZZZ": 10}))
    res = _run(r)
    assert len(calls) >= 2, "the last look never happened"
    assert res.decided is True, "the decision itself should still stand"
    assert b.sent == [], "submitted after the operator halted mid-session"


def test_without_a_reader_the_constructed_lifecycle_is_still_used():
    """Backtests and tests pass a value object and no reader; that path must keep working."""
    r, _, b = _runner(state=State.SHADOW)
    assert _run(r).decided is True
    assert b.sent == []


# -- ownership: a strategy may only see, and only sell, WHAT IT OWNS --------------------------------
def test_held_symbols_are_THIS_STRATEGYS_not_the_whole_account():
    """TECHIVOL-005's FIRST LIVE DECISION LISTED EXITS FOR THE ENTIRE BCTROT + MOMENTUM BOOK.

    Observed 2026-08-21 by kumo-trading-platform: it proposed exiting AEM, AMGN, BDX, BETA, CGAU, WHD, WPM and
    XLV — eight names, none of which it owns. Nothing reached the broker that session, so this cost
    nothing; had it submitted, it would have flattened two other strategies' positions.

    `NautilusBroker` deliberately offers BOTH:

        positions()           every open position the node knows about
        strategy_positions()  only what THIS strategy holds, per Nautilus's NETTING attribution

    This gateway called the first. QC345's calls `strategy_positions()` in all four of its ownership
    reads. The docstring on `strategy_positions` names this exact failure: "If MOMENTUM opened 10 AAPL
    and a manual trade holds 90, an exit submitted SELL 100 -- liquidating someone else's position to
    flatten ours."
    """
    r, _, b = _runner(brk=Brk(positions={"FOREIGN1": 10, "FOREIGN2": 5}))
    b.strategy_positions = lambda: {}          # this strategy owns nothing
    assert asyncio.run(r._held_symbols()) == [], (
        "read the whole account as its own holdings — it would propose exiting names it does not own")


def test_a_SELL_is_sized_to_OUR_holding_not_the_accounts():
    """The dangerous half. `_qty` for a SELL read `broker.positions()`, so an exit would be submitted
    for the FULL ACCOUNT quantity of a symbol another strategy also holds — selling their shares to
    flatten ours."""
    r, _, b = _runner(brk=Brk(positions={"SHARED": 100}))
    b.strategy_positions = lambda: {"SHARED": 10}       # we own 10 of the account's 100
    assert r._qty("SHARED", "SELL", None) == 10, "sized the exit against the whole account"


def test_no_ownership_read_in_this_gateway_uses_account_level_positions():
    """AIMED AT THE CLASS. Both defects above were one method call each, and a third would look just
    as reasonable. Asserted over the AST rather than the text, because this module explains the unsafe
    alternative in prose and a substring check would be satisfied by deleting the explanation."""
    import ast
    import inspect

    from kumo_strategies.strategies.qc27_tech_inverse_vol import runner as qc27_runner
    tree = ast.parse(inspect.getsource(qc27_runner))
    bad = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "positions"
           and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "broker"]
    assert not bad, (
        f"`broker.positions()` is the ACCOUNT's book, not this strategy's — use "
        f"`strategy_positions()`. Offending lines: {bad}")


# -- every broker call a runner makes must EXIST on the broker --------------------------------------
def test_the_runner_only_calls_broker_methods_that_are_DECLARED():
    """QC27 CALLED `broker.price()`, WHICH DOES NOT EXIST — a second, independent reason TECHIVOL-005
    sized every entry to zero.

    `NautilusBroker` has `last_price`, never `price`. `_qty` wraps the call in `except Exception:
    return 0`, so the AttributeError became "0 shares" and the journal read `sizing yielded 0 shares
    at 492.73` — a price the caller had already fetched, which is exactly why it read like a sizing
    bug rather than a missing method.

    I fixed the OTHER cause this morning (`broker_equity` reading a key cockpit never publishes) and
    the symptom would have persisted, because there were two.

    ASSERTED AGAINST THE PROTOCOL, not against NautilusBroker: the runner is written to `Broker`, and
    anything it calls that the protocol does not declare is an assumption no test on either side of
    the seam checks. That is the shape of the property-vs-method break, the Quantity-vs-float break,
    and the missing `strategy_positions` — every one a double more forgiving than production.
    """
    import ast
    import inspect

    from kumo_strategies.strategies.qc27_tech_inverse_vol import runner as qc27_runner
    from kumo_strategies.runtime.executor.broker import Broker

    declared = {n for n in dir(Broker) if not n.startswith("_")}
    used = set()
    for node in ast.walk(ast.parse(inspect.getsource(qc27_runner))):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                and node.value.attr in ("broker", "_broker")):
            used.add(node.attr)
    undeclared = sorted(used - declared)
    assert not undeclared, (
        f"calls broker methods the Broker protocol does not declare: {undeclared}. Either the "
        f"protocol is missing them or the runner is calling something that does not exist — "
        f"`price` was the latter and cost every entry.")


# -- an EXIT must go through exit(), which releases the shares --------------------------------------
def test_a_SELL_goes_through_exit_not_submit():
    """A PLAIN `submit()` SELL IS STRUCTURALLY REFUSED WHEN A PROTECTIVE STOP RESTS ON THE POSITION.

    `NautilusBroker.exit()` calls cockpit's `release_for_exit`: it suppresses protection, cancels the
    specific resting stop, and WAITS for the venue to confirm the shares are free. `submit()` does none
    of that. Its own docstring records the failure and the incident — "available: 0 means unreserved is
    0, not held is 0. FSM and VCTR both hit this" (issue 48, kumo-trading-platform issue 358).

    This runner was written AFTER that was documented and sent both sides through `submit()`.

    IT IS NOT CONDITIONAL TODAY. Every held symbol at Alpaca currently shows `qty_available = 0` —
    100% reserved by resting stops:

        AEM 18/0/2   AMGN 8/0/2   BDX 65/0/2   BETA 79/0/1   CGAU 174/0/2   WPM 26/0/2

    So a SELL that does not cancel the stop first gets `403 insufficient qty available (available: 0)`,
    every time, for reasons unrelated to the REJECTED-cache deadlock.
    """
    import asyncio

    calls = []

    class B(Brk):
        def submit(self, req):
            calls.append(("submit", req.side, req.symbol))
            return super().submit(req)

        async def exit(self, req):
            calls.append(("exit", req.side, req.symbol))
            return super().submit(req)

    r, _, b = _runner(brk=B(positions={"ZZZ": 10}))
    _run(r)
    sells = [c for c in calls if c[1] == "SELL"]
    assert sells, "no SELL formed — the fixture cannot distinguish the two paths"
    assert all(c[0] == "exit" for c in sells), (
        f"a SELL went through submit() instead of exit(), so no resting stop is cancelled and the "
        f"venue refuses it on reserved shares: {sells}")


def test_a_BUY_still_goes_through_submit():
    """`exit()` releases protection; routing an ENTRY through it would suppress protection on a
    position being opened."""
    import asyncio

    calls = []

    class B(Brk):
        def submit(self, req):
            calls.append(("submit", req.side))
            return super().submit(req)

        async def exit(self, req):
            calls.append(("exit", req.side))
            return super().submit(req)

    r, _, b = _runner(brk=B())
    _run(r)
    buys = [c for c in calls if c[1] == "BUY"]
    assert buys, "no BUY formed"
    assert all(c[0] == "submit" for c in buys), f"an ENTRY was routed through exit(): {buys}"


# -- the slot the caller named is the slot the journal stores -------------------------------------

def _rebalance_session() -> str:
    """A session that IS a rebalance date for the live cadence, so the run reaches the journal."""
    from kumo_strategies.strategies.qc27_tech_inverse_vol import rebalance_dates

    return str(sorted(rebalance_dates(PANEL["date"], LIVE_CFG.rebalance_period))[-1].date())


def test_the_slot_ARGUMENT_reaches_every_journal_row():
    """The runner accepted `slot=` and then filed every row under a module constant.

    Measured live by kumo-trading-platform on 2026-08-22: settings said `TECHIVOL-005_SLOTS = ['open+315m']`,
    cockpit honoured it for the OFFSET, and the journal reported `open+150m` for every row — the
    runner's `DECISION_SLOT` default. The lane fired at one time and named another.

    A PARAMETER ACCEPTED AND DISCARDED IS WORSE THAN A MISSING ONE. A missing parameter is a
    `TypeError` at the call; an ignored one is a caller that looks wired, a signature that looks
    complete, and a column that is confidently wrong. `exec_action_log.slot` is also half of the
    `(strategy_id, session, slot)` unique index, so the constant is the idempotency key too: two
    slots in one session would collide on it rather than being told apart.
    """
    r, j, _ = _runner()
    session = _rebalance_session()
    asyncio.run(r.run(PANEL, session, slot="open+315m"))
    assert j.rows, "the runner journalled nothing at all"
    wrong = sorted({s for s in j.slots if s != "open+315m"})
    assert not wrong, (
        f"rows filed under {wrong} while the caller named 'open+315m'; the argument is accepted and "
        f"discarded")


def test_the_IDEMPOTENCY_check_asks_about_the_slot_it_was_given():
    """`decided_this_session` is the restart guard, and it was asking about the constant.

    Its consequence is the opposite of the labelling one and worse: with the constant, a second slot
    in the same session asks "has open+150m decided?" — a question about a slot that never ran — so
    the guard either blocks a decision that should happen or permits one that should not, depending
    on which slot got there first.
    """
    r, j, _ = _runner()
    asyncio.run(r.run(PANEL, _rebalance_session(), slot="close-20m"))
    assert j.asked_slots == ["close-20m"], (
        f"the restart guard asked about {j.asked_slots}, not the slot it was handed")


def test_the_default_still_applies_when_no_slot_is_named():
    """Reading the argument must not break the single-slot caller that passes nothing."""
    r, j, _ = _runner()
    asyncio.run(r.run(PANEL, _rebalance_session()))
    assert set(j.slots) == {DECISION_SLOT}, f"default caller now files under {set(j.slots)}"


# -- the lane must CLAIM what it holds (kumo-trading-platform issue 540) -------------------------------------------
def test_an_accepted_entry_RECORDS_the_claim_and_an_exit_waits_for_terminal_sync():
    """TECHIVOL-005 never wrote an `exec_position_state` row, so it contributed nothing to
    `pgrunner._foreign_claims` — and `own_ceiling(acct, mine, other)` narrows sizing by that term.
    On alpaca-paper 2026-08-25 DELL was QC345 7 + TECHIVOL 2 with neither claiming, so any lane that
    DID claim DELL computed `own_ceiling(acct, mine, 0)` = `mine`: the attribution term vanished
    exactly when there were two other holders.

    Bound to `_send` on the real runner, with the store's writers captured — a test that called
    `record_claim` directly would prove the function works and not that the lane calls it, which is
    the entire defect.
    """
    import kumo_strategies.runtime.executor.store as store

    recorded, dropped = [], []

    async def _rec(journal, sid, sym, qty, px):
        recorded.append((sid, sym, qty, px))

    async def _drop(journal, sid, sym):
        dropped.append((sid, sym))

    orig_rec, orig_drop = store.record_claim, store.drop_claim
    store.record_claim, store.drop_claim = _rec, _drop
    try:
        r, _, b = _runner(brk=Brk(positions={"AAA": 5}), limits=RiskLimits(allocated_equity=20_000.0))
        asyncio.run(r._send("AAA", "BUY", "2026-08-25", notional=1_000.0))
        asyncio.run(r._send("AAA", "SELL", "2026-08-25"))
    finally:
        store.record_claim, store.drop_claim = orig_rec, orig_drop

    assert recorded and recorded[0][1] == "AAA", f"an accepted entry recorded no claim: {recorded}"
    assert recorded[0][0] == "TECHIVOL-005", "the claim was filed under the wrong lane"
    assert recorded[0][3] == 100.0, "the claim's entry must be the price the order was sized at"
    assert dropped == [], (
        f"accepted submit is not a venue fill; dropping here orphans a live position if the exit is "
        f"rejected: {dropped}")


def test_a_REFUSED_order_claims_NOTHING():
    """`ok` gates the write. Claiming a refused order would make every other lane size around a
    position this one never opened — and on a shared symbol that silently shrinks their ceiling."""
    import kumo_strategies.runtime.executor.store as store

    recorded = []

    async def _rec(journal, sid, sym, qty, px):
        recorded.append(sym)

    orig = store.record_claim
    store.record_claim = _rec
    try:
        r, _, b = _runner(brk=Brk(ok=False, detail="refused"),
                          limits=RiskLimits(allocated_equity=20_000.0))
        asyncio.run(r._send("AAA", "BUY", "2026-08-25", notional=1_000.0))
    finally:
        store.record_claim = orig
    assert recorded == [], f"claimed a position the venue refused: {recorded}"


def test_a_claim_failure_whose_JOURNAL_ALSO_FAILS_still_does_not_raise():
    """`_write_claim`'s contract is that it NEVER raises into the order path — the order is already
    accepted at the venue by then, so a bookkeeping failure must not turn a live position into an
    exception, nor abandon a rotation with sells already away.

    The first version guarded the claim write and then handed off to a bare `print(..., flush=True)`,
    which raises `BrokenPipeError`/`OSError`/`ValueError` when stdout is closed, replaced or broken —
    so the fallback protecting the contract could violate it (2026-08-26). There was also no
    test forcing BOTH writes to fail, which is exactly why the residual was invisible.

    This drives the real `_send` with every reporting path broken at once.
    """
    import kumo_strategies.runtime.executor.store as store

    async def _boom_claim(journal, sid, sym, qty, px):
        raise RuntimeError("ledger unavailable")

    class _DeadJrn(Jrn):
        async def write(self, *a, **k):
            if any("claim not recorded" in str(x) for x in a):
                raise RuntimeError("journal unavailable")
            return 1

    class _DeadOut:
        def write(self, *a, **k):
            raise BrokenPipeError("stdout is gone")

        def flush(self):
            raise BrokenPipeError("stdout is gone")

    orig = store.record_claim
    store.record_claim = _boom_claim
    old_stdout = sys.stdout
    sys.stdout = _DeadOut()
    try:
        r, _, b = _runner(jrn=_DeadJrn(), limits=RiskLimits(allocated_equity=20_000.0))
        ok, detail = asyncio.run(r._send("AAA", "BUY", "2026-08-25", notional=1_000.0))
    finally:
        sys.stdout = old_stdout
        store.record_claim = orig

    assert ok is True, "the order was accepted at the venue; a bookkeeping failure must not undo that"
    assert b.sent, "the order never reached the broker"


# ==================================================================================================
# THE DAILY-LOSS ROUND TRIP — producer and consumer bound through two real sessions
# ==================================================================================================

def test_the_anchor_written_in_one_session_HALTS_the_next():
    """NOTHING IS INJECTED. Session one writes its own equity anchor through `run()`; session two
    reads it back through `run()` and halts on the fall.

    THE ONLY SHAPE THAT CATCHES THE WIRING DEFECT. qc27's DECISION basis had no `equity` key, so
    `daily_loss.baseline` would have found nothing on every session forever — a halt that looks
    wired and can never fire. Every AST test still passed, and a fixture that injected
    `{"detail": {"equity": ...}}` into `tail` (which the pgrunner fixture legitimately does) would
    have passed too, over a dead mechanism. A double more forgiving than production hides exactly
    this.
    """
    from kumo_strategies.strategies.qc27_tech_inverse_vol import rebalance_dates
    reb = sorted(rebalance_dates(PANEL["date"], LIVE_CFG.rebalance_period))
    assert len(reb) >= 2, "need two rebalance sessions to write an anchor and then read it"
    s1, s2 = str(reb[-2].date()), str(reb[-1].date())

    j, halts = Jrn(), []

    class Lc(Lifecycle):
        def halt(self, why):
            halts.append(why)

    r = QC27SessionRunner(journal=j, lifecycle=Lc(State.TRADING, "test"), cfg=LIVE_CFG,
                          broker=Brk(equity=100_000.0), limits=RiskLimits(daily_loss_frac=0.05),
                          daily_loss_armed=True)
    asyncio.run(r.run(PANEL, s1))
    anchors = [d.get("equity") for k, _, d in j.decisions if k == DECISION and "equity" in d]
    assert anchors == [100_000.0], (
        f"session one wrote no equity anchor, so nothing downstream can measure a loss: {anchors}")

    r2 = QC27SessionRunner(journal=j, lifecycle=Lc(State.TRADING, "test"), cfg=LIVE_CFG,
                           broker=Brk(equity=50_000.0), limits=RiskLimits(daily_loss_frac=0.05),
                           daily_loss_armed=True)
    res = asyncio.run(r2.run(PANEL, s2))
    assert halts, (
        "a 50% fall against the anchor session one actually wrote did not halt — the producer and "
        "the consumer disagree about where the anchor lives (kumo-trading-platform issue 548)")
    assert res.blocked and "daily loss" in res.blocked


def test_the_lane_is_UNARMED_by_default_and_says_so():
    """`RiskLimits.daily_loss_frac` defaults to 0.05 and cockpit builds this runner with a bare
    `RiskLimits()`. Without `daily_loss_armed`, wiring the stop would have armed TECHIVOL-005 at 5%
    on deploy — making the operator's decision silently, on a SHARED account where another lane's
    drawdown trips it (kumo-trading-platform issue 517).

    Unarmed must not mean silent: that is the whole complaint of #548.
    """
    j = Jrn()
    r, _, _ = _runner(jrn=j, brk=Brk(equity=1_000.0), limits=RiskLimits(daily_loss_frac=0.05))
    assert r.daily_loss_armed is False, "the lane armed itself"
    _run(r)
    said = " ".join(s for _, s, _ in j.rows)
    assert "NOT ARMED" in said, (
        "an unarmed lane wrote nothing about it, so 'had no reason to halt' and 'has no ability "
        "to' are still indistinguishable — the exact complaint of kumo-trading-platform issue 548")


def test_a_HALT_reaches_the_PERSISTENCE_seam_not_just_the_local_object():
    """`_read_state` ignores `self.lifecycle` entirely when cockpit supplies a reader, so a local
    `lifecycle.halt()` survives one session and evaporates — a durable stop degraded to a one-session
    decline, on the runner being newly wired."""
    persisted = []

    async def on_halt(reason):
        persisted.append(reason)

    j = Jrn()
    j.decisions.append((DECISION, "2020-01-01", {"equity": 100_000.0}))
    r = QC27SessionRunner(journal=j, lifecycle=Lifecycle(State.TRADING, "t"), cfg=LIVE_CFG,
                          broker=Brk(equity=10_000.0), limits=RiskLimits(daily_loss_frac=0.05),
                          daily_loss_armed=True, on_halt=on_halt)
    _run(r)
    assert persisted, (
        "the halt never reached the persistence seam, so cockpit's stored state still says TRADING "
        "and the lane resumes tomorrow into the same drawdown")
