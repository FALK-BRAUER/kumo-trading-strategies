"""A position the runner refuses to SELL is still a position it HOLDS, and entries wait for the
exits that fund them (#224; platform issue 1058, #194, #1006).

TWO DEFECTS, ONE RUNNER, both rotation lanes, measured on an Alpaca paper instance (platform issue 1058):

  (A) `pgrunner` zeroes the sellable quantity of a claim the venue attributes to nobody — right:
      a strategy may never sell another strategy's shares — and then DROPS the symbol from the
      held set it hands `decide()`. The engine sees a free slot and enters. BCTROT-004 journaled
      `hold 9 · enter 4 · exit 0` at n_hold 8, and the venue answered "holds +9 AEM; a BUY 9 adds
      to it". Refusing to size the exit is right; refusing to count the position is the defect.

  (B) `_submit` sends every exit, then every entry, with no dependency on whether the exits went
      out. A refused release ("shares not released — protection still holds them") journals an
      ERROR and the buys go anyway — into cockpit's budget gate, which is blind to in-flight sells
      (#1006) and rejects them "0 of budget left". Rotation stalls on the exit side and reports it
      as a budget problem. The exit funds the entry, or the entry waits.

Driven through the REAL `PgSessionRunner.run()` on a rankable panel, the fixture shape
`test_gap_uses_todays_open.py` established. Every test here was seen red on 1f5fe2e for the reason
its docstring names.
"""

from __future__ import annotations

import asyncio as aio

import pandas as pd
import pytest

from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
from kumo_strategies.strategies.momentum_rotation.config import (
    MomentumRotationConfig, PortfolioConfig)

SESSION = "2026-08-04"


def _panel(scores: dict[str, float]) -> pd.DataFrame:
    """80 rising sessions per name; `scores` sets each name's daily drift so the RANKING is chosen
    by the test, not by noise. Higher drift ranks higher."""
    dates = pd.bdate_range("2026-04-14", periods=80)
    rows = []
    for t, drift in scores.items():
        px = 100.0
        for i, d in enumerate(dates):
            px = px * (1.0 + (drift if i % 3 else -0.001))
            rows.append({"ticker": t, "date": d, "open": px * 0.999, "high": px * 1.01,
                         "low": px * 0.99, "close": px, "volume": 5_000_000})
    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


class _Jrn:
    def __init__(self):
        self.rows: list[tuple[str, str, dict]] = []

    async def write(self, kind, summary, **kw):
        self.rows.append((kind, summary, kw))
        return len(self.rows)

    async def decided_this_session(self, session, slot=None):
        return False

    async def tail(self, n=400, kind=None, symbol=None):
        return []

    def summaries(self, kind=None):
        return [s for k, s, _ in self.rows if kind is None or k == kind]


def _runner(*, cfg, account: dict, claims: dict, attributed: dict | None, sent: list,
            refuse_release: set[str] = frozenset(), jrn: _Jrn | None = None,
            foreign: dict | None = None):
    jrn = jrn or _Jrn()

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"A", "B", "C", "D"}

        async def must_liquidate(self, sym):
            return False

    class Broker:
        def positions(self):
            return dict(account)

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 100.0

        def submit(self, req):
            sent.append(req)
            return type("R", (), {"ok": True, "detail": "accepted"})()

        async def exit(self, req):
            if req.symbol in refuse_release:
                # `NautilusBroker.exit`, broker.py:372-373, when cockpit's release never comes.
                return type("R", (), {"ok": False, "detail": "shares not released — protection "
                                                                "still holds them"})()
            sent.append(req)
            return type("R", (), {"ok": True, "detail": "accepted"})()

    if attributed is not None:
        Broker.strategy_positions = lambda self: dict(attributed)

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            return dict(foreign or {})

        async def _owned(self):
            return dict(claims)

        async def _reconcile(self, *a, **kw):
            return None

        async def _load_state(self, *a, **kw):
            return {}

        async def _save_state(self, *a, **kw):
            return None

        async def _drop_state(self, sym):
            return None

    r = Runner(strategy_id="BCTROT-004", pool=Pool(), journal=jrn,
               lifecycle=Lifecycle(State.TRADING, "armed"), cfg=cfg, broker=Broker())
    return r, jrn


def _cfg(n_hold: int = 2) -> MomentumRotationConfig:
    return MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=n_hold, buffer=0))


def _buys(sent) -> list[str]:
    return sorted(r.symbol for r in sent if r.side == "BUY")


def _sells(sent) -> list[str]:
    return sorted(r.symbol for r in sent if r.side == "SELL")


def _decision(jrn: _Jrn) -> dict:
    rows = [kw for k, _, kw in jrn.rows if k == "decision"]
    assert rows, f"no decision row: {jrn.summaries()}"
    return rows[-1]


# ==================================================================================================
# (A) held is not sellable
# ==================================================================================================

#: A and B rank top-2; C and D below. The book holds A and B: a full book at n_hold 2.
_AB_TOP = {"A": 0.006, "B": 0.005, "C": 0.003, "D": 0.002}


def test_FIXTURE_a_full_attributed_book_at_n_hold_neither_enters_nor_exits():
    """Fixture property first: when the venue attributes both names, the runner sees a full book
    and sends nothing. If this were red, the (A) test below would be measuring the fixture."""
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10, "B": 10}, sent=sent)
    aio.run(r.run(_panel(_AB_TOP), SESSION, slot="open+5m"))
    assert _buys(sent) == [] and _sells(sent) == [], sent
    assert sorted(_decision(jrn)["detail"]["target_book"]) == ["A", "B"]


def test_a_claimed_position_the_venue_does_NOT_attribute_still_OCCUPIES_its_slot():
    """THE (A) DEFECT. B is claimed and on the account, the venue attributes only A. On 1f5fe2e the
    runner zeroes B's sellable size (correct), drops B from the held set, and the engine — seeing
    one free slot in a top-2 book whose #2 is B — sends a BUY for B: the AEM row, verbatim shape
    ("holds +9 AEM; a BUY 9 adds to it"). A position we cannot sell is a position we hold."""
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10}, sent=sent)
    aio.run(r.run(_panel(_AB_TOP), SESSION, slot="open+5m"))
    assert _buys(sent) == [], (
        f"the runner bought into a full book: {_buys(sent)} — the unattributed claim was counted "
        f"as a free slot (#224 A)")
    d = _decision(jrn)["detail"]
    assert sorted(d["target_book"]) == ["A", "B"], d["target_book"]
    # The refusal to SIZE B's exit is unchanged and still said out loud.
    assert any("venue attributes it NONE" in s for s in jrn.summaries("risk")), jrn.summaries("risk")


def test_the_unsellable_position_is_HELD_for_the_ranking_but_NOT_sellable_for_the_exit():
    """The split, both halves on one fixture: C and D now outrank A and B, so the engine rotates
    both out. A (attributed) is SOLD; B (claimed, unattributed) is counted as held, is NOT sold
    because it cannot be sized — and because its exit did not go out, the entries it was meant to
    fund WAIT (#224 B applies to an unsizable exit exactly as to a refused release). On 1f5fe2e
    both C and D were bought on top of B: a three-name book at n_hold 2."""
    sent: list = []
    cd_top = {"C": 0.006, "D": 0.005, "A": 0.003, "B": 0.002}
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10}, sent=sent)
    aio.run(r.run(_panel(cd_top), SESSION, slot="open+5m"))
    assert _sells(sent) == ["A"], f"sold something it cannot size, or nothing at all: {_sells(sent)}"
    assert _buys(sent) == [], (
        f"entries were sent while B's exit could not go out: {_buys(sent)} (#224 A+B)")
    held = [s for s in jrn.summaries("risk") if "entries held" in s]
    assert len(held) == 1 and "B" in held[0] and "cannot be sized" in held[0], jrn.summaries("risk")


def test_the_journaled_hold_count_NEVER_exceeds_n_hold_and_INCLUDES_the_unsellable_name():
    """platform issue 1058 acceptance 1, on the decision row: `target_book` is the book the lane HOLDS
    after the session, unsellable names included, and it is never longer than n_hold."""
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10}, sent=sent)
    aio.run(r.run(_panel(_AB_TOP), SESSION, slot="open+5m"))
    book = _decision(jrn)["detail"]["target_book"]
    assert "B" in book and len(book) <= 2, book


def test_TWO_claimants_and_no_attribution_is_held_but_not_sellable():
    """Sibling (coverage review): B claimed 10 by us AND 10 by another lane, the account holds 10,
    no attribution at all — `own_ceiling(10, 10, 10)` is 0, and on 1f5fe2e that dropped B from the
    held set exactly like the unattributed case (the GMAB 59+59 shape on paper 09-09). B occupies
    a slot; nothing is bought; B is not sold."""
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed=None, sent=sent, foreign={"B": 10.0})
    aio.run(r.run(_panel(_AB_TOP), SESSION, slot="open+5m"))
    assert _buys(sent) == [] and _sells(sent) == [], sent
    assert sorted(_decision(jrn)["detail"]["target_book"]) == ["A", "B"]


def test_a_STALE_claim_not_on_the_account_counts_NEITHER_as_held_nor_as_sellable():
    """Sibling: a claim on a name the account no longer holds is retired at session start — it must
    not occupy a slot either, or the fix re-creates the stale-claim slot leak the other way. With A
    held and the stale B gone, one slot is free and the top-ranked free name is bought."""
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10}, sent=sent)
    aio.run(r.run(_panel(_AB_TOP), SESSION, slot="open+5m"))
    assert _buys(sent) == ["B"], f"the stale claim still occupied a slot: {_buys(sent)}"
    assert "B" not in [r_.symbol for r_ in sent if r_.side == "SELL"]


def test_a_SHORT_account_quantity_on_a_long_lane_is_neither_held_nor_sellable_and_is_SAID():
    """Sibling: a short on the account (reconciled in, manual, phantom — this lane opens none) must
    not count as a slot the rotation occupies, must never be sold by it, and must be reported.
    Measured while writing this: the claim on a short is RETIRED at session-start reconcile
    ("released 1 position claims this strategy does not hold"), so `held`'s `acct_qty > 0` guard is
    belt-and-braces — a mutant dropping it is equivalent on this path, and that is stated rather
    than a green read as coverage."""
    sent: list = []
    # A > C > B > D: with B NOT a slot the free slot goes to C; a runner counting the short as held
    # would see a full book and buy nothing.
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": -10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10}, sent=sent)
    aio.run(r.run(_panel({"A": 0.006, "C": 0.005, "B": 0.003, "D": 0.002}), SESSION, slot="open+5m"))
    assert "B" not in _sells(sent), "a long-only lane sized a SELL against a short"
    assert any("SHORT positions" in s_ for s_ in jrn.summaries("risk")), jrn.summaries("risk")
    assert _buys(sent) == ["C"], f"the short was counted as an occupied slot: {_buys(sent)}"


# ==================================================================================================
# (B) entries wait for the exits that fund them
# ==================================================================================================

#: C and D outrank A and B: the session rotates A, B out and C, D in.
_CD_TOP = {"C": 0.006, "D": 0.005, "A": 0.003, "B": 0.002}


def _rotating(sent, refuse: set[str]):
    return _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                   attributed={"A": 10, "B": 10}, sent=sent, refuse_release=refuse)


def test_FIXTURE_when_every_exit_goes_out_the_entries_follow_as_today():
    """Fixture property: with both sells accepted the buys are sent — the (B) test below must be
    red because of the REFUSAL, not because entries never go."""
    sent: list = []
    r, jrn = _rotating(sent, refuse=set())
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))
    assert _sells(sent) == ["A", "B"] and _buys(sent) == ["C", "D"], sent


def test_when_ONE_exit_is_refused_NO_entry_is_sent_that_slot_and_the_journal_says_why():
    """THE (B) DEFECT. A's release never comes. On 1f5fe2e the runner journals the ERROR and sends
    both buys anyway — which cockpit's budget gate then rejects "0 of budget left" (#1006), and the
    stall reads as a budget problem. The exit funds the entry, or the entry waits."""
    sent: list = []
    r, jrn = _rotating(sent, refuse={"A"})
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))
    assert _sells(sent) == ["B"], sent
    assert _buys(sent) == [], (
        f"entries were sent in a slot where an exit was refused: {_buys(sent)} (#224 B)")
    held = [s for s in jrn.summaries("risk") if "entries held" in s]
    assert len(held) == 1, jrn.summaries("risk")
    assert "A" in held[0] and "not released" in held[0], held[0]


def test_when_EVERY_exit_is_refused_NO_entry_is_sent_either():
    sent: list = []
    r, jrn = _rotating(sent, refuse={"A", "B"})
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))
    assert _sells(sent) == [] and _buys(sent) == [], sent


def test_a_held_entry_is_NOT_marked_attempted_so_the_next_slot_can_send_it_after_the_sell():
    """The wait must be a WAIT, not a refusal: `_resume` treats a symbol as attempted only on a
    `phase=result` row (or a suppression row), so the held entries must leave neither — otherwise
    the resume path would never send them once the sell goes through next slot. Read off the rows
    the runner wrote: no result row and no `suppressed` detail for C or D."""
    sent: list = []
    r, jrn = _rotating(sent, refuse={"A"})
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))
    for _, _, kw in jrn.rows:
        d = kw.get("detail") or {}
        if kw.get("symbol") in ("C", "D"):
            assert d.get("phase") != "result", kw
            assert not d.get("suppressed"), kw
    assert not any(s for s in jrn.summaries() if "suppressed" in s and ("C" in s or "D" in s))


def test_the_resume_path_sends_the_waiting_entries_once_the_sell_goes_through():
    """Same session, next slot: the sell is accepted now, so `_submit` — the method `_resume` calls
    with the not-yet-attempted enters and exits — sends the exit AND the entries it was holding."""
    sent: list = []
    r, jrn = _rotating(sent, refuse=set())
    n = aio.run(r._submit(SESSION, ("C", "D"), ("A",), {"A": 10, "B": 10}, {}, State.TRADING,
                          decision_slot="open+150m"))
    assert _sells(sent) == ["A"] and _buys(sent) == ["C", "D"] and n == 3, sent


def _held_row(session: str, refused: dict, entries: list, slot: str = "open+5m") -> dict:
    """A journal row the runner itself writes when it holds entries — the shape `held_sessions_before`
    reads. Built here as the runner writes it (kind RISK, detail keys), not hand-typed prose."""
    return {"kind": "risk", "session": session, "slot": slot,
            "summary": "entries held this slot", "detail": {
                "refused_exits": refused, "held_entries": entries, "held_count": len(entries),
                "held_sessions_before": 0, "slot": slot}}


def _journal_with_tail(rows: list) -> _Jrn:
    """A tail that honours `kind` and `n` the way PgJournal.tail does (newest first, limited), so a
    streak read that asked for too few rows of the wrong kind is caught here, not on paper."""
    j = _Jrn()

    async def tail(n=400, kind=None, symbol=None):
        picked = [r for r in rows if kind is None or r.get("kind") == kind]
        return list(reversed(picked))[:n]
    j.tail = tail
    return j


def _pool_noise(session: str, n: int = 200) -> list[dict]:
    """What a busy session writes beside its RISK rows: BCTROT wrote ~45 pool rows per session on
    paper 09-11; 200 here so a 400-row any-kind tail cannot reach two sessions back."""
    return [{"kind": "pool", "session": session, "summary": f"pool row {i}", "detail": {}}
            for i in range(n)]


def test_the_cap_TWO_held_sessions_then_the_THIRD_enters_anyway_with_a_loud_row():
    """The lead's ruling on (B): entries wait the rest of the session; capped at 2 consecutive
    SESSIONS with the same refused exits; on the 3rd the lane enters anyway and says what is still
    refused. Unbounded never-enter is a silently shrinking book with every surface green."""
    refused = {"A": "shares not released — protection still holds them"}
    prior = [_held_row("2026-07-31", refused, ["C", "D"]), _held_row("2026-08-03", refused, ["C", "D"])]
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10, "B": 10}, sent=sent, refuse_release={"A"},
                     jrn=_journal_with_tail(prior))
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))       # SESSION = 2026-08-04, the 3rd
    assert _buys(sent) == ["C", "D"], f"the cap did not release the entries: {_buys(sent)}"
    released = [s_ for s_ in jrn.summaries("risk") if "entries released after 2" in s_]
    assert len(released) == 1 and "A" in released[0] and "still refused" in released[0], jrn.summaries("risk")
    assert not any("entries held" in s_ for s_ in jrn.summaries("risk"))


def test_the_streak_is_read_by_KIND_so_a_busy_session_cannot_hide_a_prior_held_session():
    """Impl review: a 400-row any-kind tail on a lane writing hundreds of pool rows per session
    misses yesterday's held row, the streak reads 0 forever, and the cap silently becomes the
    unbounded rule it exists to refuse. 200 pool rows per session on each side; the 2-session
    streak must still be seen and the 3rd session released."""
    refused = {"A": "shares not released — protection still holds them"}
    prior = (_pool_noise("2026-07-31") + [_held_row("2026-07-31", refused, ["C", "D"])]
             + _pool_noise("2026-08-03") + [_held_row("2026-08-03", refused, ["C", "D"])]
             + _pool_noise(SESSION))
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10, "B": 10}, sent=sent, refuse_release={"A"},
                     jrn=_journal_with_tail(prior))
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))
    assert _buys(sent) == ["C", "D"], f"the streak did not see through the pool rows: {_buys(sent)}"


def test_the_refused_exits_detail_names_a_CAUSE_per_symbol():
    """The operator's fix is nameable from the journal: `release_refused` (cockpit's protection),
    `unsizable` (reconcile the claim), `intent_not_journalled` (the database), without the log."""
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10}, sent=sent, refuse_release={"A"})
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))
    rows = [kw["detail"] for k, s_, kw in jrn.rows if k == "risk" and "entries held" in s_]
    assert len(rows) == 1
    causes = {s2: v["cause"] for s2, v in rows[0]["refused_exits"].items()}
    assert causes == {"A": "release_refused", "B": "unsizable"}, causes


def test_the_cap_counts_SESSIONS_not_slots_ONE_held_session_of_three_slots_is_one():
    """Three slots held yesterday are ONE held session: today is the second, still held."""
    refused = {"A": "shares not released — protection still holds them"}
    prior = [_held_row("2026-08-03", refused, ["C", "D"], slot=sl)
             for sl in ("open+5m", "open+150m", "close-20m")]
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10, "B": 10}, sent=sent, refuse_release={"A"},
                     jrn=_journal_with_tail(prior))
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))
    assert _buys(sent) == [], f"slots were counted as sessions: {_buys(sent)}"
    held = [s_ for s_ in jrn.summaries("risk") if "entries held" in s_]
    assert len(held) == 1 and "session 2 of 2" in held[0], held


def test_a_held_row_from_an_EARLIER_SLOT_TODAY_is_not_a_prior_session():
    """Today's own earlier slots do not count toward the cap: one prior session + today's first
    slot held → session 2 of 2 at today's second slot, still held, not released."""
    refused = {"A": "shares not released — protection still holds them"}
    prior = [_held_row("2026-08-03", refused, ["C", "D"]),
             _held_row(SESSION, refused, ["C", "D"], slot="open+5m")]
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10, "B": 10}, sent=sent, refuse_release={"A"},
                     jrn=_journal_with_tail(prior))
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+150m"))
    assert _buys(sent) == [], f"today's earlier slot was counted as a prior session: {_buys(sent)}"
    assert any("session 2 of 2" in s_ for s_ in jrn.summaries("risk")), jrn.summaries("risk")


def test_a_DIFFERENT_refused_exit_restarts_the_streak():
    """The cap is for the SAME refused exits. Two sessions held for A, today B is the one refused:
    session 1 of 2 for B — held, not released."""
    prior = [_held_row("2026-07-31", {"A": "x"}, ["C", "D"]), _held_row("2026-08-03", {"A": "x"}, ["C", "D"])]
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10, "B": 10}, sent=sent, refuse_release={"B"},
                     jrn=_journal_with_tail(prior))
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))
    assert _buys(sent) == [], _buys(sent)
    assert any("session 1 of 2" in s_ for s_ in jrn.summaries("risk")), jrn.summaries("risk")


def test_the_held_row_carries_COUNTS_and_the_slot_in_detail_for_the_readback():
    """platform issue 1058's readback sums rather than parses: `held_count`, `slot`, `refused_exits` and
    `held_entries` travel in `detail`, and the release row carries the same shape under `released_*`."""
    sent: list = []
    r, jrn = _rotating(sent, refuse={"A"})
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))
    rows = [kw["detail"] for k, s_, kw in jrn.rows if k == "risk" and "entries held" in s_]
    assert len(rows) == 1
    d = rows[0]
    assert d["held_count"] == 2 and d["slot"] == "open+5m" and d["held_entries"] == ["C", "D"]
    assert set(d["refused_exits"]) == {"A"} and d["held_sessions_before"] == 0


def test_resume_with_NO_refused_exit_still_sends_an_unsent_entry():
    """Mutant (c) from the review: a resume that treated 'no ok result row' as unfunded for
    ENTRIES would hold them forever. No exits owed, one entry unsent → it goes."""
    sent: list = []
    r, jrn = _rotating(sent, refuse=set())
    n = aio.run(r._submit(SESSION, ("C",), (), {"A": 10, "B": 10}, {}, State.TRADING,
                          decision_slot="open+150m"))
    assert _buys(sent) == ["C"] and n == 1, sent


def test_LIQUIDATING_is_unaffected_it_has_no_entries_to_hold():
    """The urgent path submits exits only; a refused release there must still journal its ERROR
    and hold nothing back, because there is nothing to hold."""
    sent: list = []
    r, jrn = _runner(cfg=_cfg(2), account={"A": 10, "B": 10}, claims={"A": 10, "B": 10},
                     attributed={"A": 10, "B": 10}, sent=sent, refuse_release={"A"})
    r.lifecycle = Lifecycle(State.LIQUIDATING, "operator")
    aio.run(r.run(_panel(_CD_TOP), SESSION, slot="open+5m"))
    assert _sells(sent) == ["B"] and _buys(sent) == [], sent
    assert not any("entries held" in s for s in jrn.summaries("risk")), jrn.summaries("risk")
