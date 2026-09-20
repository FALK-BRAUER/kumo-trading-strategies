"""#124 step-9 review: `ClaimWriteStalled` reaches `_save_state`/`_drop_state` call sites in
`pgrunner`, and only ONE of them is the fail-closed pre-submit save the docstring describes.

    :586   reconcile-time adoption loop        — a stall must not abandon the remaining orphans
    :666   retire loop, before the exit submit — a stall must not stop the remaining retirements/exits
    :1028  the pre-submit trail save           — FAIL-CLOSED, deliberately: abort before money moves

Bookkeeping that stalls after the venue has answered costs a journal row, not the session.
"""
from __future__ import annotations

import asyncio as aio

import pytest

from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
from kumo_strategies.runtime.executor.runner import RiskLimits
from kumo_strategies.runtime.executor.store import ClaimWriteStalled
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig
from kumo_strategies.strategies.momentum_rotation.exits import TrailState


class _Broker:
    def __init__(self):
        self.sent = []

    def submit(self, req):
        from kumo_strategies.runtime.executor.broker import OrderResult
        self.sent.append(req)
        return OrderResult(True, "id", "accepted", req)

    async def exit(self, req):
        return self.submit(req)

    def positions(self):
        return {}

    def equity(self):
        return 100_000.0

    def last_price(self, sym, **_):
        return 10.0


class _Journal:
    strategy_id = "MOMENTUM-002"

    def __init__(self):
        self.rows = []

    async def write(self, kind, summary, **kw):
        self.rows.append((kind, summary))
        return len(self.rows)

    async def tail(self, n=400, kind=None, symbol=None):
        return []


def _runner(journal, broker, limits=None):
    return PgSessionRunner(strategy_id="MOMENTUM-002", pool=None, journal=journal,
                           lifecycle=Lifecycle(State.TRADING, "armed"), cfg=MomentumRotationConfig(),
                           broker=broker, limits=limits or RiskLimits(book_size=8))


def _stall_once(monkeypatch, r, method: str, on_symbol: str):
    calls = []

    async def _stalling(sym, *a, **k):
        calls.append(sym)
        if sym == on_symbol:
            raise ClaimWriteStalled(f"MOMENTUM-002/{sym}: Postgres did not finish the claim write within 30s")

    monkeypatch.setattr(r, method, _stalling)
    return calls


def test_a_locally_accepted_BUY_does_NOT_write_a_claim_before_terminal_fill(monkeypatch):
    """2026-09-14: `ok=True` meant Nautilus accepted the order locally, then the budget gate denied
    it. Writing the claim at submit-time left CRAK/DINO claimed with no position."""
    j, b = _Journal(), _Broker()
    r = _runner(j, b)
    calls = _stall_once(monkeypatch, r, "_save_state", on_symbol="AAA")
    aio.run(r._submit("2026-09-10", ("AAA", "BBB"), (), {}, {"AAA": 10.0, "BBB": 10.0}, State.TRADING))
    assert [q.symbol for q in b.sent] == ["AAA", "BBB"], [q.symbol for q in b.sent]
    assert calls == [], "submit-time acceptance wrote a durable claim before terminal fill"


def test_a_locally_accepted_BUY_still_occupies_capacity_inside_the_submit_loop(monkeypatch):
    """Removing the submit-time claim must not let one session overbook itself while entries are
    still in flight."""
    j, b = _Journal(), _Broker()
    r = _runner(j, b, limits=RiskLimits(book_size=1, max_positions=2))
    calls = _stall_once(monkeypatch, r, "_save_state", on_symbol="AAA")

    aio.run(r._submit("2026-09-10", ("AAA", "BBB"), (), {"HELD": 1},
                      {"AAA": 10.0, "BBB": 10.0}, State.TRADING))

    assert [q.symbol for q in b.sent] == ["AAA"], [q.symbol for q in b.sent]
    assert calls == [], "capacity should be in-memory until the terminal event syncs the claim"


def test_a_stall_in_the_RETIRE_loop_does_not_stop_the_remaining_retirements(monkeypatch):
    """`:666` — `for sym in stale_claims: await self._drop_state(sym)`, immediately before the exit
    submit. A stall on the first must not leave the second claim standing and the exits unsent."""
    j, b = _Journal(), _Broker()
    r = _runner(j, b)
    calls = _stall_once(monkeypatch, r, "_drop_state", on_symbol="AAA")

    async def _drop_all():
        # Drive the loop body as `_reconcile` runs it (pgrunner.py ~:664-667).
        for sym in ("AAA", "BBB"):
            await r._retire_claim(sym, session="2026-09-10")

    aio.run(_drop_all())
    assert calls == ["AAA", "BBB"], calls
    assert any("AAA" in s and "stall" in s.lower() for k, s in j.rows), j.rows


def test_the_PRE_SUBMIT_trail_save_stays_FAIL_CLOSED(monkeypatch):
    """`:1028` — the one site the docstring describes. A trail the database would not take is not a
    trail to size orders against; the stall propagates and nothing is submitted."""
    j, b = _Journal(), _Broker()
    r = _runner(j, b)
    _stall_once(monkeypatch, r, "_save_state", on_symbol="AAA")
    r._pending_trail = {"AAA": (TrailState(entry_px=10.0, peak_px=10.0), 5)}
    with pytest.raises(ClaimWriteStalled):
        aio.run(r._persist_pending_trail())
    assert b.sent == []
