"""The gap dead band must fire on TODAY's gap, driven through the real `PgSessionRunner.run`.

This file exists because every other test of the rule passed while it was computing the WRONG DAY's
gap. `test_gap_dead_band.py` binds the AST — that the live path assigns `enters` from
`filter_entries_by_gap`, and that the second argument mentions "open". Neither can see WHICH date's
open, and the defect was entirely in the date.

The mechanism, for the next person: `_try_decide` hands the runner
`panel[panel.date <= prior]` where `prior` is the last session STRICTLY BEFORE the one being
decided. That trim is correct — today's prices must not vote on today's ranking — but it means the
panel's newest row is YESTERDAY. Reading `day["open"]` from it therefore gave yesterday's open, and
the prior close it was divided by was the day BEFORE yesterday. The rule fired on the previous
session's gap, at all three slots, and journalled "gap filter declined N entries" while doing it.

So today's open is passed in by the caller instead — from `self._bars`, where Alpaca's republished
forming daily bar for today actually lives — and the prior close is `last`, the newest close in the
trimmed panel, which IS yesterday's close.

Behavioural, not structural: a flat-gap name must not be submitted and a gapped one must be. Both
assertions were seen RED against the previous implementation before this file was kept.
"""

from __future__ import annotations

import asyncio as aio

import pandas as pd

from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
from kumo_strategies.strategies.momentum_rotation.config import (
    ExecutionConfig, MomentumRotationConfig, PortfolioConfig)

# D-2 and D-1 are what the runner's panel contains. D is today: never in the panel, only in `opens`.
#
#            D-2 close   D-1 open   D-1 close   D open    today's gap    YESTERDAY's gap
#   FLAT        100.0      101.0      100.0      100.2      +0.20%          +1.00%
#   GAPPER      100.0      100.2      100.0      104.0      +4.00%          +0.20%
#
# Chosen so the two days DISAGREE in both directions: reading yesterday's gap admits FLAT and
# declines GAPPER — the exact inversion of what the rule is for.
def _panel() -> pd.DataFrame:
    """80 sessions of warmup, then two crafted days.

    The warmup is not decoration: with a short panel every score is NaN, the runner correctly
    refuses to decide, and the test passes or fails for reasons having nothing to do with the gap.
    A rising series with small noise gives both names a real, rankable momentum score.

    The last two sessions are built so TODAY and YESTERDAY disagree in OPPOSITE directions:

              D-2 close  D-1 open  D-1 close  D open   today's gap  yesterday's gap
        FLAT      100.0     105.0      100.0    100.2     +0.20% IN     +5.00% OUT
        GAPPER    100.0     100.5      100.0    104.0     +4.00% OUT    +0.50% IN

    D-2's close is pinned too, not left to the generated series — otherwise "yesterday's gap" is
    whatever the warmup happened to produce and only one of the two assertions discriminates.

    Both names invert under the defect: reading yesterday ADMITS the flat one and DECLINES the
    gapped one, so either test alone catches it and neither can pass by accident.
    """
    dates = pd.bdate_range("2026-04-14", periods=80)
    rows = []
    for t in ("FLAT", "GAPPER"):
        px = 100.0
        for i, d in enumerate(dates):
            px = px * (1.0 + (0.004 if i % 3 else -0.002))
            rows.append({"ticker": t, "date": d, "open": px * 0.999, "high": px * 1.01,
                         "low": px * 0.99, "close": px, "volume": 5_000_000})
    df = pd.DataFrame(rows)
    # The last TWO sessions are pinned: D-1 is the newest row the runner may see, and D-2's close
    # is what a wrong implementation would divide D-1's open by.
    y, y2 = dates[-1], dates[-2]
    df.loc[(df.ticker == "FLAT") & (df.date == y2), "close"] = 100.0
    df.loc[(df.ticker == "GAPPER") & (df.date == y2), "close"] = 100.0
    df.loc[(df.ticker == "FLAT") & (df.date == y), ["open", "close"]] = [105.0, 100.0]
    df.loc[(df.ticker == "GAPPER") & (df.date == y), ["open", "close"]] = [100.5, 100.0]
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


PANEL = _panel()
SESSION = "2026-08-04"          # the day AFTER the panel's newest row
TODAYS_OPENS = {"FLAT": 100.2, "GAPPER": 104.0}


def _runner(cfg: MomentumRotationConfig, sent: list):
    class Jrn:
        """Enforces `uq_exec_one_decision_per_session` the way Postgres does (#831).

        The double this replaced returned 1 for every write, so it could not represent the partial
        unique index `(strategy_id, session, slot) WHERE kind='decision'` — and BCTROT-004 journalled
        "gap filter declined N entries" under kind=decision, took the index slot, and never decided
        for five consecutive slots while this file stayed green. A double must reject what
        production rejects.
        """
        def __init__(self):
            self.rows = []
            self.decided = set()

        async def write(self, kind, summary, **kw):
            from kumo_strategies.runtime.executor.pgjournal import DuplicateDecision
            key = (kw.get("session"), kw.get("slot"))
            if kind == "decision":
                if key in self.decided:
                    raise DuplicateDecision(f"already decided {key}")
                self.decided.add(key)
            self.rows.append((kind, summary, kw))
            return len(self.rows)

        async def decided_this_session(self, session, slot=None):
            return False

        async def tail(self, n=400, kind=None, symbol=None):
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"FLAT", "GAPPER"}

        async def must_liquidate(self, sym):
            return False

    class Broker:
        def positions(self):
            return {}

        def equity(self):
            return 100_000.0

        def last_price(self, sym):
            return TODAYS_OPENS[sym]

        def submit(self, req):
            # SYNC, deliberately: pgrunner calls `self.broker.submit(...)` without awaiting while it
            # DOES await `broker.exit(...)`. A double that made both async silently produced a
            # coroutine whose `.ok` was an attribute error, and no order.
            sent.append(req)
            return type("R", (), {"ok": True, "detail": "filled"})()

        async def exit(self, req):
            sent.append(req)
            return type("R", (), {"ok": True, "detail": "filled"})()

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            return {}

        async def _owned(self):
            # No claims ledger in these fixtures. Overridden rather than faked on the journal,
            # because a fake `sessionmaker` would be a second thing to keep honest and this test is
            # about which DATE the gap reads, not about ownership.
            return {}

        async def _reconcile(self, *a, **kw):
            return None

        async def _load_state(self, *a, **kw):
            return {}

        async def _save_state(self, *a, **kw):
            # Trail persistence needs a real sessionmaker. Stubbed rather than faked for the same
            # reason as `_owned`: this test is about which DATE the gap reads.
            return None

    return Runner(strategy_id="BCTROT-004", pool=Pool(), journal=Jrn(),
                  lifecycle=Lifecycle(State.TRADING, "armed"), cfg=cfg, broker=Broker())


def _armed() -> MomentumRotationConfig:
    return MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=2, buffer=0),
        execution=ExecutionConfig(min_abs_gap_pct=0.015))


def _bought(sent) -> set[str]:
    return {r.symbol for r in sent if getattr(r, "side", "") == "BUY"}


def test_a_flat_gap_TODAY_is_declined_even_though_YESTERDAY_gapped():
    """FLAT gapped +5.00% yesterday and +0.20% today. Today's is inside the band so it must be
    declined; an implementation reading yesterday sees +5.00%, outside, and buys it."""
    sent: list = []
    r = _runner(_armed(), sent)
    aio.run(r.run(PANEL, SESSION, slot="open+5m", opens=TODAYS_OPENS))
    assert "FLAT" not in _bought(sent), (
        f"FLAT's overnight gap is +0.20%, inside the 1.5% dead band, and it was bought: {sent}")


def test_a_REAL_gap_TODAY_is_admitted_even_though_YESTERDAY_was_flat():
    """GAPPER opened +4.00% today and only +0.50% yesterday. Today's is outside the band so it must
    be bought; an implementation reading yesterday sees +0.50%, inside, and declines it — the rule
    rejecting precisely the name it exists to keep."""
    sent: list = []
    r = _runner(_armed(), sent)
    aio.run(r.run(PANEL, SESSION, slot="open+5m", opens=TODAYS_OPENS))
    assert "GAPPER" in _bought(sent), (
        f"GAPPER's overnight gap is +4.00%, outside the band, and it was NOT bought — the rule is "
        f"reading the wrong day: {sent}")


def test_the_rule_admits_everything_when_no_opens_are_supplied():
    """Fails open, and this is the state a feed that has not yet republished today's bar produces.
    Safe, but silent — `_decide` journals a RISK line naming the blind symbols so that 'inert this
    slot' is distinguishable from 'nothing to decline'."""
    sent: list = []
    r = _runner(_armed(), sent)
    aio.run(r.run(PANEL, SESSION, slot="open+5m", opens={}))
    assert {"FLAT", "GAPPER"} <= _bought(sent), (
        f"with no opens the rule must admit rather than decline: {sent}")


def test_a_declined_entry_does_not_consume_the_decision_slot():
    """#831, measured on an Alpaca paper instance 2026-09-04 → 2026-09-09: every BCTROT-004 slot with ≥1 declined
    name ended `NO DECISION — already decided … (concurrent run)` and the lane never decided again.
    The decline row was journalled under kind=decision and the unique index took it for THE decision.

    Fixture property first: the run declines something (FLAT is inside the band). Then the real
    decision row must exist and the session must not report itself blocked."""
    sent: list = []
    r = _runner(_armed(), sent)
    res = aio.run(r.run(PANEL, SESSION, slot="open+5m", opens=TODAYS_OPENS))
    rows = r.journal.rows
    declined = [x for x in rows if "gap filter declined" in x[1]]
    assert declined, f"fixture cannot express the bug — nothing was declined: {rows}"
    decisions = [x for x in rows if x[0] == "decision" and str(x[2].get("correlation", "")).endswith(":decision")]
    assert decisions, (
        f"no decision row was written — the decline row took the one-decision-per-slot index: "
        f"blocked={res.blocked!r} rows={[(k, m) for k, m, _ in rows]}")
    assert res.blocked is None, f"the session reported itself blocked: {res.blocked!r}"
    assert "GAPPER" in _bought(sent), "the decision did not submit — GAPPER should have been bought"


def test_an_unarmed_config_buys_both_regardless():
    """The control. Without it, a rule that declined EVERYTHING would pass the first test."""
    sent: list = []
    r = _runner(MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=2, buffer=0)), sent)
    aio.run(r.run(PANEL, SESSION, slot="open+5m", opens=TODAYS_OPENS))
    assert {"FLAT", "GAPPER"} <= _bought(sent), f"unarmed config did not buy both: {sent}"


# -- size_to_book: WHICH book the denominator counts ---------------------------------------------
# Latent today (cockpit pins size_to_book=False) but untested until now: a mutation replacing the
# post-decision book with the PRE-decision one passed every test in tests/runtime/executor. That is
# the book `sizing_denominator`'s own docstring warns against, because it still contains names being
# sold this session and would size every entry too small.


def _sized(cfg, held, sent) -> dict[str, int]:
    r = _runner(cfg, sent)

    class Broker:
        def positions(self):
            return dict(held)

        def equity(self):
            return 100_000.0

        def last_price(self, sym):
            return TODAYS_OPENS.get(sym, 100.0)

        def submit(self, req):
            sent.append(req)
            return type("R", (), {"ok": True, "detail": "filled"})()

        async def exit(self, req):
            sent.append(req)
            return type("R", (), {"ok": True, "detail": "filled"})()

    r.broker = Broker()
    aio.run(r.run(PANEL, SESSION, slot="open+5m", opens=TODAYS_OPENS))
    return {req.symbol: req.qty for req in sent if getattr(req, "side", "") == "BUY"}


def test_size_to_book_counts_the_POST_decision_book_not_the_pre_decision_one():
    """Two names enter into an empty book, so the post-decision book is 2 and the pre-decision book
    is 0. Sizing off the pre-decision book falls back to the PLANNED book (`book_size`), which is a
    different, smaller slot — so the quantities move.

    Asserting the RATIO rather than absolute share counts: the point is which denominator was used,
    and absolute quantities depend on equity, deployed fraction and price in ways this test should
    not pin.
    """
    on = MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=2, buffer=0, size_to_book=True),
        execution=ExecutionConfig())
    off = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=2, buffer=0),
                                 execution=ExecutionConfig())

    sent_on: list = []
    sent_off: list = []
    qty_on = _sized(on, {}, sent_on)
    qty_off = _sized(off, {}, sent_off)

    assert qty_on and qty_off, f"nothing was bought: on={sent_on} off={sent_off}"
    # book_size defaults to 8 while the actual book is 2, so sizing to the book must buy MORE.
    for sym in qty_off:
        assert qty_on[sym] > qty_off[sym], (
            f"{sym}: size_to_book bought {qty_on[sym]} against {qty_off[sym]} unsized — the "
            f"denominator is not the book that exists. A pre-decision book here is 0, which falls "
            f"back to the planned book and reproduces the OFF quantities exactly")
