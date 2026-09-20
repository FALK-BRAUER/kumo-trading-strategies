"""The daily-loss halt must not be disarmed by an equity that is not a number.

FOUND BY AUDIT 2026-08-28, not by a failure — which is the point: a disarmed halt does nothing
visible until the day it was needed.

    pgrunner.py:522   now = self.broker.equity()          <- unguarded
    pgrunner.py:535   if start and now < start * (1 - self.limits.daily_loss_frac):

`now` is used in a comparison with no check that it is a finite number, and there are two ways it
is not:

    NaN    every comparison with a NaN is False, so `now < threshold` is False and the halt is
           SKIPPED. Silently. The strategy keeps trading through the limit it was configured to
           respect.
    None   `None < float` raises TypeError, out of the session decision path.

NEITHER IS HYPOTHETICAL. `NautilusBroker.equity()` reads the msgbus account snapshot, which is
absent until the first account update — that is the None. And cockpit found `Decimal("NaN")` coming
off their Alpaca account handling on 2026-08-28, which is the NaN.

THIS EXACT DEFECT WAS ALREADY FIXED ONCE, in the other runner:

    qc27_runner.py:286   return value if math.isfinite(value) else 0.0

`qc27_runner` guards; `pgrunner` does not. Two implementations of one rule, and only one of them
was repaired — the shape this repo keeps paying for. pgrunner is MOMENTUM-002 and BCTROT-004, which
are the lanes actually trading on the paper tenant.

No test covered the daily-loss halt at all before this file.
"""

from __future__ import annotations

import asyncio

import pandas as pd

import pytest

from kumo_strategies.runtime.executor.pgjournal import DECISION
from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
from kumo_strategies.runtime.executor.runner import RiskLimits
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig


_SESSION = "2026-08-28"
#: The baseline the journal reports for the previous session. A 20% fall from here must halt.
_BASELINE = 100_000.0


def _fixture(equity, *, baseline=_BASELINE, frac=0.05):
    """A runner whose only interesting variable is what `broker.equity()` returns."""
    halts, rows = [], []

    class Broker:
        def submit(self, req):
            from kumo_strategies.runtime.executor.broker import OrderResult
            return OrderResult(True, "id", "detail", req)

        async def exit(self, req):
            return self.submit(req)

        def positions(self):
            return {}

        def equity(self):
            return equity

        def last_price(self, sym, **_):
            return 10.0

    class Jrn:
        strategy_id = "MOMENTUM-002"
        sessionmaker = None

        async def write(self, kind, summary, **kw):
            rows.append((kind, summary, kw))
            return 1

        async def decided_this_session(self, *a, **k):
            return False

        async def tail(self, n=400, kind=None, symbol=None):
            # A REAL prior decision carrying equity, so the baseline is found the way production
            # finds it rather than injected past the lookup.
            if kind == DECISION:
                return [{"session": "2026-08-27", "detail": {"equity": baseline}}]
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"AAA"}

        async def must_liquidate(self, sym):
            return False

    class Lc(Lifecycle):
        def halt(self, why):
            halts.append(why)

    class Runner(PgSessionRunner):
        # Persistence is stubbed so the halt is the ONLY thing under test. Everything else returns
        # the shape production returns, empty — not a looser shape that would hide a defect.
        async def _save_state(self, sym, st, qty=None):
            return None

        async def _drop_state(self, sym):
            return None

        async def _load_state(self, symbols):
            return {}

        async def _owned(self):
            return {}

        async def _foreign_claims(self):
            return {}

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(),
               lifecycle=Lc(State.TRADING, "armed"), cfg=MomentumRotationConfig(),
               broker=Broker(), limits=RiskLimits(daily_loss_frac=frac))
    return r, halts, rows


def _panel() -> pd.DataFrame:
    """The smallest panel `run` will accept. The halt is checked before ranking matters."""
    days = pd.date_range("2026-06-01", periods=260, freq="B")
    return pd.DataFrame([{"ticker": "AAA", "date": d, "open": 10.0, "high": 10.0,
                          "low": 10.0, "close": 10.0, "volume": 1e6} for d in days])


def _run(r):
    return asyncio.run(r.run(_panel(), _SESSION))


def _halted(equity, **kw) -> bool:
    """Did the halt fire for this equity? Drives the REAL method, not a reimplementation."""
    r, halts, _ = _fixture(equity, **kw)
    _run(r)
    return bool(halts)


def _blocked(equity, **kw):
    """(blocked reason, was it permanently halted) — the two are a deliberate distinction."""
    r, halts, _ = _fixture(equity, **kw)
    res = _run(r)
    return (res.blocked or ""), bool(halts)


# ==================================================================================================
# THE CONTROL — without this the assertions below prove nothing
# ==================================================================================================

def test_a_real_breach_DOES_halt():
    """FIXTURE PROPERTY FIRST. If the halt never fired for any input, every assertion below would
    pass against a limit that does nothing at all."""
    assert _halted(80_000.0), (
        "a 20% fall against a 5% limit did not halt — the fixture never reaches the check, so the "
        "NaN and None cases below would be vacuous")


def test_a_normal_session_does_NOT_halt():
    """The other half of the control. A halt that fires on everything is equally useless."""
    assert not _halted(99_000.0), "a 1% fall halted against a 5% limit"


# ==================================================================================================
# THE DEFECT
# ==================================================================================================

def test_a_NaN_equity_does_not_DISARM_the_halt():
    """MEASURED RED. `NaN < threshold` is False, so the comparison passes and the halt is skipped —
    the strategy keeps trading through the limit, silently.

    A NaN reaches here from a `Decimal("NaN")` in the account snapshot, which cockpit measured on
    their Alpaca account handling on 2026-08-28.
    """
    blocked, halted = _blocked(float("nan"))
    assert blocked, (
        "an equity of NaN skipped the daily-loss check entirely. Every comparison with a NaN is "
        "False, so the limit silently does not apply and the strategy keeps trading")
    assert not halted, (
        "a NaN equity triggered the PERMANENT halt. A missing account snapshot is transient and "
        "clears itself; halting on it needs an operator to un-halt a lane that was never in breach")


def test_a_MISSING_equity_does_not_crash_the_session():
    """`NautilusBroker.equity()` reads the msgbus account snapshot, which is absent until the first
    account update — so None is the ordinary startup value, not an exotic one.

    `None < float` raises TypeError out of the session decision path.
    """
    r, halts, _ = _fixture(None)
    try:
        res = _run(r)
    except TypeError as exc:
        pytest.fail(f"a missing equity raised out of the session path: {exc!r}")
    assert res.blocked, "an unreadable equity was treated as a healthy one"
    assert not halts, "a transient missing snapshot must not need an operator to clear"


def test_an_unreadable_equity_is_REPORTED_not_just_handled():
    """Halting is not enough. "halted on a 20% loss" and "halted because we cannot read the account"
    are different operator problems, and a halt that does not say which sends someone hunting a
    drawdown that never happened."""
    r, halts, rows = _fixture(float("nan"))
    res = _run(r)
    said = " ".join(str(s) for _, s, _ in rows) + " " + (res.blocked or "")
    assert any(w in said.lower() for w in ("not a number", "unreadable", "cannot be read", "nan")), (
        f"nothing said the equity was unreadable; an operator sees only a loss figure. Got: {said}")


# ==================================================================================================
# THE OTHER RUNNER ALREADY GUARDS THIS — the two must not disagree
# ==================================================================================================

def test_both_runners_guard_the_equity_they_compare_against():
    """Two derivations of one rule, and only one was repaired.

    `qc27_runner._account_equity` has used `math.isfinite` since the NaN sizing defect. `pgrunner`
    reads `broker.equity()` straight into a comparison. Bound to the AST rather than a substring:
    a grep is satisfied by a comment naming `isfinite` and broken by one that does not.
    """
    import ast
    import inspect

    from kumo_strategies.strategies.momentum_rotation import runner as pgrunner
    from kumo_strategies.strategies.qc27_tech_inverse_vol import runner as qc27_runner

    for mod in (pgrunner, qc27_runner):
        tree = ast.parse(inspect.getsource(mod))
        # SCOPED TO THE EQUITY READER, not the module. The first version of this assertion looked
        # for `isfinite` ANYWHERE in the file and passed against unguarded pgrunner — it has
        # `np.isfinite` on prices at :443 and :1213, which says nothing about the account value.
        # A check that finds the right word in the wrong function is not a check.
        readers = [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and "equity" in n.name]
        assert readers, f"{mod.__name__} has no named equity reader to guard"
        guards = {a.attr for r in readers for a in ast.walk(r)
                  if isinstance(a, ast.Attribute) and a.attr == "isfinite"}
        assert guards, (
            f"{mod.__name__} performs no finiteness check anywhere. It compares an account value "
            f"against a risk limit, and a NaN makes every such comparison False — which disarms the "
            f"limit rather than tripping it.")


# ==================================================================================================
# THE SAME READ, ONE FUNCTION AWAY — sizing (2026-08-28, before BCTROT's first IBKR decision)
# ==================================================================================================

def _sizing_fixture(equity):
    """A runner that will ENTER, so the sizing read is reached. Exits are given too, because the
    property that matters is that they still go out."""
    r, halts, rows = _fixture(equity, baseline=None)
    sent = []
    real_submit = r.broker.submit

    def _spy(req):
        sent.append((req.side, req.symbol))
        return real_submit(req)

    r.broker.submit = _spy

    async def _exit(req):
        return _spy(req)

    r.broker.exit = _exit
    return r, sent, rows


def test_sizing_refuses_rather_than_half_executing_on_a_NaN_equity():
    """MEASURED: `pgrunner.py:1229` read `broker.equity()` straight into the arithmetic.

    A NaN makes `slot` NaN, survives every downstream comparison — all of them are False against a
    NaN — and raises `cannot convert float NaN to integer` inside the quantity conversion PART WAY
    THROUGH the entry loop, after earlier symbols have already been submitted. A half-executed
    rotation is worse than none.

    MOMENTUM-002 and BCTROT-004 pass a bare `RiskLimits()`, so `allocated_equity` is None and this
    IS their sizing path.
    """
    r, sent, rows = _sizing_fixture(float("nan"))
    try:
        asyncio.run(r._submit(_SESSION, ("AAA",), (), {}, {"AAA": 10.0}, State.TRADING))
    except (ValueError, TypeError) as exc:
        pytest.fail(f"a NaN equity reached the arithmetic and raised: {exc!r}")
    assert not [s for s in sent if s[0] == "BUY"], "an entry was sized off a NaN equity"
    assert any("cannot size" in str(s).lower() for _, s, _ in rows), (
        "no entries were submitted and nothing said why")


def test_a_MISSING_equity_does_not_raise_out_of_the_decision_path():
    """`None * frac` raises TypeError. `NautilusBroker.equity()` reads the msgbus account snapshot,
    absent until the first account update — the ordinary state of a node that has just connected,
    which is exactly BCTROT's first decision on IBKR."""
    r, sent, _ = _sizing_fixture(None)
    try:
        asyncio.run(r._submit(_SESSION, ("AAA",), (), {}, {"AAA": 10.0}, State.TRADING))
    except (ValueError, TypeError) as exc:
        pytest.fail(f"a missing equity raised out of the decision path: {exc!r}")
    assert not [s for s in sent if s[0] == "BUY"]


def test_an_unreadable_equity_STOPS_US_BUYING_NEVER_SELLING():
    """THE PROPERTY THAT MUST HOLD. This function already states the rule for an unhonourable exit
    config — stop us buying, never selling — and the refusal has to obey it. Exits run before the
    sizing read, so refusing there must not swallow them."""
    r, sent, _ = _sizing_fixture(float("nan"))
    asyncio.run(r._submit(_SESSION, ("AAA",), ("HELD",), {"HELD": 5}, {"AAA": 10.0, "HELD": 20.0},
                          State.TRADING))
    assert ("SELL", "HELD") in sent, (
        "an unreadable equity blocked an EXIT. Sizing is an entry concern; a position we already "
        "hold must still be sellable")
    assert not [s for s in sent if s[0] == "BUY"]


def test_a_REAL_equity_still_sizes_and_buys():
    """The control. A refusal that fired on every input would pass all three assertions above."""
    r, sent, _ = _sizing_fixture(100_000.0)
    asyncio.run(r._submit(_SESSION, ("AAA",), (), {}, {"AAA": 10.0}, State.TRADING))
    assert [s for s in sent if s[0] == "BUY"], "nothing was bought on a perfectly readable equity"


# ==================================================================================================
# THE OTHER SIDE OF THE COMPARISON — the RECORDED baseline (#111)
# ==================================================================================================
# Everything above varies `now`. `baseline` was only ever a good number or None, so the axis was
# never probed — and a mutation bite can only reach an axis some fixture varies. The defect lived
# there for the whole life of both lanes that HAVE this stop.
#
# MEASURED RED before the fix, through this same `_blocked`:
#
#     good baseline, 99% fall  ->  halted=True
#     NaN  baseline, 99% fall  ->  halted=False, blocked='only 0/1 candidates carry a score...'
#
# The session walked past the risk check and died later on an unrelated ranking floor, which in the
# journal is indistinguishable from an ordinary refusal. A control that fails by continuing.

def test_a_NaN_BASELINE_is_LOUD_even_though_it_proceeds():
    """THE #111 HEADLINE, corrected. The defect was never "it proceeded" — it is that it proceeded
    SILENTLY, so a disarmed stop looked exactly like a working one.

    `_session_start_equity` returned `float(NaN)` because `.get("equity")` sees a NaN as truthy, and
    `now < NaN * 0.95` is False like every comparison with a NaN. The session then died on an
    unrelated ranking floor, which in the journal reads as an ordinary refusal.

    Refusing outright was my first fix and it was wrong — see `test_a_poisoned_journal_does_not_
    DEADLOCK`. What has to be true is that the lane SAYS it is running unenforceable.
    """
    r, halts, rows = _fixture(1_000.0, baseline=float("nan"))
    _run(r)
    said = " ".join(str(s) for _, s, _ in rows)
    assert "UNENFORCEABLE" in said, (
        f"a 99% fall against a poisoned anchor was not evaluated and the lane said nothing about "
        f"it — a disarmed stop indistinguishable from a working one (#111). Rows: {said[:300]}")
    assert "non-finite" in said, "the row does not say WHY the stop could not be evaluated"
    assert not halts, (
        "a poisoned anchor triggered the PERMANENT halt. A broken past write is not a breach.")


def test_a_poisoned_journal_does_not_DEADLOCK():
    """WHY REFUSING IS NOT THE ANSWER, kept as a test because I shipped the wrong one first.

    A BLOCK returns before the DECISION write, so a blocked session records no new anchor. The next
    session scans back, finds the same non-finite rows, and blocks again — forever, with nothing an
    operator can do, because nothing rewrites history. That is a permanent stop wearing transient
    clothes, and it inverts the BLOCK/HALT distinction the module exists to keep.
    """
    for _ in range(3):
        blocked, halted = _blocked(1_000.0, baseline=float("nan"))
        assert not (blocked and "unusable" in blocked), (
            "a poisoned anchor blocks the session, and a blocked session writes no new anchor — so "
            "this repeats every session forever. Measured block/block/block before the fix.")
        assert not halted


def test_a_NaN_baseline_and_a_NaN_equity_are_told_APART():
    """Both refuse, and they must not say the same thing. One means the account is unreadable NOW,
    the other means a previous session recorded garbage as this lane's anchor — different
    investigations, and collapsing them sends an operator to the wrong one."""
    now_bad, _ = _blocked(float("nan"))
    assert now_bad and "unverifiable" in now_bad, (
        f"an unreadable equity TODAY must still refuse the session — the next session re-reads the "
        f"live account and heals, so this one does not deadlock: {now_bad!r}")
    r, _, rows = _fixture(1_000.0, baseline=float("nan"))
    _run(r)
    assert "UNENFORCEABLE" in " ".join(str(s) for _, s, _ in rows), (
        "a poisoned ANCHOR must proceed-and-say-so, never share the refusal an unreadable equity "
        "gets — one is a fact about today's value, the other about a past write")


def test_a_GOOD_baseline_still_halts_after_all_this():
    """VACUITY GUARD. Every assertion above is about refusing. If the fix made the runner refuse
    everything, they would all pass while the stop no longer worked."""
    assert _halted(80_000.0), "the ordinary breach stopped halting"
