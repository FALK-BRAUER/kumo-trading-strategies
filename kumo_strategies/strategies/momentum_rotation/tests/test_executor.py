"""Tests for the runtime. Each pins a safety property, not a happy path.

Pure tests run anywhere. Anything needing the store is marked `pg` and skips without a reachable
Postgres — the point of the store tests is the real database's behaviour (transactions, the unique
index, tz-aware timestamps), which a fake would not exercise.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
import pytest


from kumo_strategies.runtime.executor import (
    EXCLUDE, PIN, Lifecycle, OrderRequest, State, TransitionRejected)

PG_URL = os.environ.get("KUMO_TEST_DATABASE_URL") or os.environ.get("KUMO_DATABASE_URL")


def _fresh_generated_at() -> str:
    """A sidecar timestamp that is current NOW.

    `LedgerBookSource` refuses an export older than MAX_META_AGE_HOURS (30) — correctly, since a stale
    book must not drive a session. Fixtures below hardcoded `2026-08-04T00:00:00Z`, so three tests
    about sidecar mismatch, row-count mismatch and expiry started failing on the staleness check
    roughly 30 hours after that date, and stayed failing. They were red for days on a clock, not on a
    defect, which is exactly how a suite stops being a signal.

    Tests that are ABOUT staleness should state their own age explicitly; everything else should use
    this and stop caring what day it is.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pg_reachable() -> bool:
    if not PG_URL:
        return False
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        async def go() -> bool:
            e = create_async_engine(PG_URL)
            try:
                async with e.connect() as c:
                    await c.execute(text("select 1"))
                return True
            finally:
                await e.dispose()
        return asyncio.run(go())
    except Exception:                                            # noqa: BLE001
        return False


pg = pytest.mark.skipif(not _pg_reachable(), reason="needs a reachable Postgres")


# -- lifecycle (pure) --------------------------------------------------------------------------
def test_nothing_automatic_may_start_trading():
    l = Lifecycle(State.SHADOW)
    with pytest.raises(TransitionRejected):
        l.to(State.TRADING, reason="auto", by_operator=False)
    l.to(State.TRADING, reason="operator", by_operator=True)
    assert l.state is State.TRADING


def test_halt_is_automatic_and_resume_is_not():
    l = Lifecycle(State.TRADING)
    l.halt("daily loss limit")
    assert l.state is State.HALTED
    with pytest.raises(TransitionRejected):
        l.to(State.TRADING, reason="straight back", by_operator=True)
    l.to(State.SHADOW, reason="reviewed", by_operator=True)


def test_shadow_decides_but_does_not_submit():
    assert State.SHADOW.decides and not State.SHADOW.may_submit_entries
    assert State.TRADING.decides and State.TRADING.may_submit_entries
    assert State.LIQUIDATING.may_submit_exits and not State.LIQUIDATING.may_submit_entries


# -- broker (pure) -----------------------------------------------------------------------------
def test_client_order_id_is_stable_per_session_and_symbol():
    a = OrderRequest("AAA", "BUY", 10, "2026-01-02", strategy_id="MOMENTUM-002")
    b = OrderRequest("AAA", "BUY", 99, "2026-01-02", strategy_id="MOMENTUM-002")           # qty must not change identity
    c = OrderRequest("AAA", "BUY", 10, "2026-01-03", strategy_id="MOMENTUM-002")
    assert a.client_order_id == b.client_order_id != c.client_order_id


def test_a_retried_attempt_gets_a_different_client_order_id():
    """Code review, not caught by any test until now: `nautilus/broker.py`'s own docstring says a
    duplicate `client_order_id` is denied LOCALLY by Nautilus — deliberate replay safety. But #51's
    retry resubmits a terminally-rejected symbol with the SAME id it used the first time, since
    nothing varied it. Nautilus would deny the resubmission as a duplicate before it ever reached the
    venue a second time — the exact scenario (VCTR) #51 exists to fix stays unfixed through the real
    broker, and the local denial journals identically to a genuine second rejection, so nothing would
    even reveal that the retry never really happened."""
    first = OrderRequest("VCTR", "SELL", 88, "2026-08-14", strategy_id="MOMENTUM-002")
    retry = OrderRequest("VCTR", "SELL", 88, "2026-08-14", strategy_id="MOMENTUM-002", attempt=1)
    assert first.client_order_id != retry.client_order_id, (
        "a retried order must not collide with the id Nautilus already denied a duplicate of")


def test_attempt_zero_is_byte_identical_to_the_original_id():
    """The overwhelming majority of orders are never retried — attempt=0 must hash to exactly what
    it did before `attempt` existed, or every first-time order silently gets a new id for no reason."""
    a = OrderRequest("AAA", "BUY", 10, "2026-01-02", strategy_id="MOMENTUM-002")
    b = OrderRequest("AAA", "BUY", 10, "2026-01-02", strategy_id="MOMENTUM-002", attempt=0)
    assert a.client_order_id == b.client_order_id


def test_a_different_decision_slot_gets_a_different_client_order_id():
    """2026-08-19 15:40 UTC, live: MOMENTUM-002's open+5m attempt for FSM this morning and its
    open+130m retry hashed to the SAME id — neither `session` nor `attempt` had changed, only the
    slot. Nautilus denied the retry as a local duplicate before it ever reached the venue, and the
    denial cost more than itself: `NautilusBroker.submit()` reads the order back from the cache BY
    this id to report the outcome, and for a duplicate that lookup returns the EARLIER order — this
    morning's, genuinely rejected by Alpaca. The journal reported a fresh venue rejection that was
    actually seven hours old, quoting a different order's real answer as though it were this one. A
    new decision slot IS a new decision — `uq_exec_one_decision_per_session` already says so, keyed
    on (strategy_id, session, slot); the order id just never agreed."""
    morning = OrderRequest("FSM", "SELL", 933, "2026-08-19", strategy_id="MOMENTUM-002", slot="open+5m")
    retry = OrderRequest("FSM", "SELL", 933, "2026-08-19", strategy_id="MOMENTUM-002", slot="open+130m")
    assert morning.client_order_id != retry.client_order_id, (
        "a retried decision under a NEW slot must not collide with an earlier slot's denied order")


def test_no_slot_is_byte_identical_to_the_original_id():
    """Every caller before this change passed no slot. `slot=""` must hash to exactly what omitting
    it did, or every existing single-slot strategy (MOMENTUM-002 today) silently gets a new id for
    no reason."""
    a = OrderRequest("AAA", "BUY", 10, "2026-01-02", strategy_id="MOMENTUM-002")
    b = OrderRequest("AAA", "BUY", 10, "2026-01-02", strategy_id="MOMENTUM-002", slot="")
    assert a.client_order_id == b.client_order_id


def test_max_price_age_seconds_defaults_off_but_actually_wires_through_when_set():
    """`None` (default) is today's behaviour: nothing in either repo ever constructs a `RiskLimits`
    with this set, so a bare `RiskLimits()` -- what both gateways actually construct -- must produce
    `{}`, not a bound nobody asked for. But the mechanism existing and doing nothing (`_max_age_kw()`
    always `{}`) is exactly how this shipped dead the first time -- so when a caller DOES set it,
    `_max_age_kw()` must actually carry it through, not silently drop it too."""
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.runtime.executor.runner import RiskLimits

    r = PgSessionRunner.__new__(PgSessionRunner)
    r.limits = RiskLimits()
    assert r._max_age_kw() == {}, "the default must stay off until proven safe on a live node"
    r.limits = RiskLimits(max_price_age_seconds=1800.0)
    assert r._max_age_kw() == {"max_age_ns": 1800_000_000_000}, (
        "a caller that DOES set the bound must have it actually reach last_price()")


def test_there_is_NO_direct_to_venue_broker_to_construct():
    """`AlpacaBroker` is deleted (the operator's ruling, 2026-08-27). It replaced this test, which only
    checked that a direct-to-venue broker refused a LIVE endpoint — a guard on the wrong question,
    since the class bypassed Nautilus entirely whichever endpoint it pointed at: no RiskEngine check,
    nothing in the cache, nothing to reconcile, and no `release_for_exit`.

    The stronger rule is enforced in `tests/runtime/test_no_direct_venue_access.py`; this pins the
    import so nothing reintroduces it under the old name."""
    import kumo_strategies.runtime.executor as ex

    assert not hasattr(ex, "AlpacaBroker"), (
        "a direct-to-venue broker is importable again — orders must go through NautilusBroker")


# -- the give-back trail must not be instance-local --------------------------------------------
def test_the_give_back_trail_must_not_live_in_memory():
    """The worst bug found in this runtime, pinned.

    The API builds a fresh session runner per request. While entry/peak lived on the instance, each
    run re-seeded them from today's price, so a position that had run +40% and was handing it back
    looked brand new and the exit NEVER FIRED. It passed every backtest, because a backtest keeps
    one runner for the whole loop.
    """
    import inspect

    from kumo_strategies.runtime.executor import PgSessionRunner
    fields = getattr(PgSessionRunner, "__dataclass_fields__", {})
    assert "_peak" not in fields and "_entry" not in fields

    # The trail is READ from the store where it is used, and WRITTEN by run() once the decision row
    # is won — deferring the write is what stops a runner that loses a concurrent race from
    # advancing state anyway. Both halves are asserted because either alone leaves it non-durable.
    assert "_load_state" in inspect.getsource(PgSessionRunner._trail_exits)
    run_src = inspect.getsource(PgSessionRunner.run)
    # The write moved into `_persist_pending_trail` (#124: the one trail write allowed to abort a
    # session) — `run` must still be the caller, once the decision row is won, and that method must
    # still be the thing that writes.
    assert "_pending_trail" in run_src and "_persist_pending_trail" in run_src
    assert "_save_state" in inspect.getsource(PgSessionRunner._persist_pending_trail)


def test_pool_departure_is_a_sell_signal_not_a_glitch():
    """Worth +62.5% vs +33.4% over 167 sessions — the pool drops a name when the followed trader
    exits it. Protection against a bad feed is upstream (stale gate, shrink guard), never here."""
    from kumo_strategies.strategies.momentum_rotation.candidates import StaticList
    from kumo_strategies.strategies.momentum_rotation.config import (
        MomentumRotationConfig, PortfolioConfig)
    from kumo_strategies.strategies.momentum_rotation.engine import (
        apply_gates, decide, score_panel)
    rows = [{"ticker": t, "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d),
             "open": px, "high": px * 1.02, "low": px * 0.98, "close": px, "volume": 5_000_000}
            for i, t in enumerate(["AAA", "BBB", "CCC"])
            for d in range(60) for px in [10.0 * (1 + 0.004 * (i + 1) * d)]]
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=2, buffer=0))
    panel = score_panel(apply_gates(pd.DataFrame(rows), cfg), cfg)
    day = panel[panel.date == panel.date.max()]
    d = decide(day, StaticList(["AAA", "BBB", "CCC"]), cfg, held={"AAA", "GONE"})
    assert "GONE" in d.exit


# -- store-backed (need Postgres) --------------------------------------------------------------
@pytest.fixture
def pool_factory():
    from kumo_strategies.runtime.executor import PgSymbolPool
    from kumo_strategies.runtime.executor.store import (
        create_all, make_engine, make_sessionmaker)
    made = []

    async def make():
        e = make_engine(PG_URL)
        await create_all(e)
        made.append(e)
        return PgSymbolPool(make_sessionmaker(e)), f"t_{uuid.uuid4().hex[:8]}"
    yield make
    for e in made:
        asyncio.run(e.dispose())


@pg
def test_pin_survives_a_source_dropping_it(pool_factory):
    async def go():
        p, src = await pool_factory()
        await p.refresh_source(src, {"AAA", "BBB"})
        await p.set_override("AAA", PIN)
        await p.refresh_source(src, {"BBB"})
        assert "AAA" in await p.symbols(), "a source must not delete what it does not own"
        await p.clear_override("AAA", PIN)
        await p.drop_source(src)
    asyncio.run(go())


@pg
def test_exclude_survives_a_source_re_adding_it(pool_factory):
    async def go():
        p, src = await pool_factory()
        await p.refresh_source(src, {"AAA"})
        await p.set_override("AAA", EXCLUDE)
        await p.refresh_source(src, {"AAA"})
        assert "AAA" not in await p.symbols(), "no whack-a-mole: an exclude must hold"
        await p.clear_override("AAA", EXCLUDE)
        await p.drop_source(src)
    asyncio.run(go())


@pg
def test_a_failed_source_keeps_its_last_good_set(pool_factory):
    async def go():
        p, src = await pool_factory()
        await p.refresh_source(src, {"AAA", "BBB"})
        await p.mark_source_failed(src, "http 500")
        assert {"AAA", "BBB"} <= await p.symbols(), "an empty pool is worse than a stale one"
        assert next(h for h in await p.sources() if h.source == src).status == "failed"
        await p.drop_source(src)
    asyncio.run(go())


@pg
def test_a_truncated_refresh_is_rejected(pool_factory):
    """The dangerous case the stale gate cannot catch: a source that SUCCEEDS while returning a
    truncated set. A pool departure is acted on as a sell signal, so that would liquidate."""
    from kumo_strategies.runtime.executor import ShrinkRejected

    async def go():
        p, src = await pool_factory()
        await p.refresh_source(src, {f"S{i}" for i in range(40)})
        with pytest.raises(ShrinkRejected):
            await p.refresh_source(src, {"S0", "S1"})
        assert len([s for s in await p.symbols() if s.startswith("S")]) >= 40
        await p.refresh_source(src, {f"S{i}" for i in range(25)})    # plausible shrink is fine
        await p.drop_source(src)
    asyncio.run(go())


@pg
def test_pool_governs_buying_only(pool_factory):
    async def go():
        p, src = await pool_factory()
        await p.refresh_source(src, {"AAA", "BBB", "CCC", "DDD"})
        await p.refresh_source(src, {"BBB", "CCC", "DDD"})           # AAA dropped, within shrink
        assert not await p.is_buyable("AAA")
        assert not await p.must_liquidate("AAA"), "a feed drop must not force liquidation"
        await p.set_override("AAA", EXCLUDE)
        assert await p.must_liquidate("AAA"), "only an operator blacklist forces a sale"
        await p.clear_override("AAA", EXCLUDE)
        await p.drop_source(src)
    asyncio.run(go())


@pg
def test_dropping_a_source_removes_its_symbols_and_health(pool_factory):
    async def go():
        p, src = await pool_factory()
        await p.refresh_source(src, {"AAA", "BBB"})
        await p.drop_source(src)
        assert not [h for h in await p.sources() if h.source == src], "orphan health row blocks"
    asyncio.run(go())


@pg
def test_two_concurrent_refreshes_of_the_same_source_do_not_collide(pool_factory):
    """ledger_book crashed at 07:15Z: two overlapping refreshes of the same source both DELETEd,
    both plain-INSERTed, and the second collided on the (source, symbol) primary key — a
    UniqueViolationError that rolled back the whole 93-row batch. A source that fails keeps its
    last good set, so this degraded rather than emptied the pool, but it also means the pool goes
    silently stale on every collision until someone notices. Two refreshes racing on the SAME
    largely-overlapping set is exactly what a scheduled refresh firing twice looks like."""
    async def go():
        p, src = await pool_factory()
        base = {f"S{i}" for i in range(50)}
        await p.refresh_source(src, base)
        a = {*base, "NEWA"}
        b = {*base, "NEWB"}
        await asyncio.gather(p.refresh_source(src, a), p.refresh_source(src, b))
        got = await p.symbols()
        assert base <= got, "a colliding refresh must not lose the shared symbols"
        assert "NEWA" in got or "NEWB" in got, "at least one refresh's own addition must survive"
        await p.drop_source(src)
    asyncio.run(go())


# -- calendar ----------------------------------------------------------------------
def test_the_calendar_knows_holidays_and_half_days():
    """A weekday heuristic fires into a closed market. The cockpit's own session guard admits it is
    holiday-unaware (engine_node.py:2743); this must not be — the broker's calendar is the venue
    that decides whether an order can fill."""
    from datetime import date
    from kumo_strategies.runtime.executor.calendar import AlpacaCalendar, WeekdayCalendar
    import os
    if not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")):
        pytest.skip("needs Alpaca credentials for the real calendar")
    c = AlpacaCalendar()
    assert not c.is_trading_day(date(2026, 11, 26)), "Thanksgiving is closed"
    assert not c.is_trading_day(date(2026, 12, 25)), "Christmas is closed"
    bf = c.day(date(2026, 11, 27))
    assert bf and bf.is_half_day, "Black Friday is a half-day"
    # and the fallback is honest about being wrong
    assert WeekdayCalendar().is_trading_day(date(2026, 11, 26)), \
        "the fallback is holiday-unaware by design; it must not pretend otherwise"


def test_the_fire_time_is_relative_to_the_open_not_the_wall_clock():
    """Offsetting from a fixed wall-clock time would fire at the wrong point of a half-day."""
    from datetime import date, datetime
    from kumo_strategies.runtime.executor.calendar import ET, WeekdayCalendar
    c = WeekdayCalendar()
    session, fire = c.next_fire(datetime(2026, 8, 3, 6, 0, tzinfo=ET), 5)
    assert fire.hour == 9 and fire.minute == 35
    assert session == date(2026, 8, 3)


# -- upstream freshness -------------------------------------------------------------------------
# Cadence answers "is our copy old?". It does not answer "has the source moved?", and those came
# apart in production: a book cached at 18:26 sat inside its 24h cadence while ledger-tool published a
# corrected file at 22:20, so the pool kept serving a symbol upstream had already fixed.

def _spec_row(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(**kw)


def test_a_source_is_due_when_upstream_is_newer_than_our_cache():
    from datetime import datetime, timezone

    from kumo_strategies.runtime.executor.pgjobs import _due

    now = datetime(2026, 8, 3, 23, 0, tzinfo=timezone.utc)
    row = _spec_row(enabled=True, every_minutes=1440, last_status="ok",
                          last_run=datetime(2026, 8, 3, 18, 26, tzinfo=timezone.utc))
    assert _due(row, now=now) is False, "cadence alone says not due — that is the bug"
    assert _due(row, now=now,
                upstream=datetime(2026, 8, 3, 22, 20, tzinfo=timezone.utc)) is True


def test_an_older_upstream_does_not_force_a_refresh():
    from datetime import datetime, timezone

    from kumo_strategies.runtime.executor.pgjobs import _due

    row = _spec_row(enabled=True, every_minutes=1440, last_status="ok",
                          last_run=datetime(2026, 8, 3, 18, 26, tzinfo=timezone.utc))
    assert _due(row, now=datetime(2026, 8, 3, 23, 0, tzinfo=timezone.utc),
                upstream=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)) is False


def test_a_disabled_source_stays_disabled_however_new_upstream_is():
    from datetime import datetime, timezone

    from kumo_strategies.runtime.executor.pgjobs import _due

    row = _spec_row(enabled=False, every_minutes=1, last_status="ok",
                          last_run=datetime(2026, 8, 3, 18, 26, tzinfo=timezone.utc))
    assert _due(row, now=datetime(2026, 8, 3, 23, 0, tzinfo=timezone.utc),
                upstream=datetime(2026, 8, 3, 22, 59, tzinfo=timezone.utc)) is False


def test_ledger_publishes_the_sidecar_generated_at(tmp_path):
    import json as _json
    from datetime import timezone

    from kumo_strategies.runtime.executor.sources.ledger import LedgerBookSource

    csv = tmp_path / "ledger-book.csv"
    csv.write_text("symbol,open_date,close_date\n")
    (tmp_path / "ledger-book.meta.json").write_text(
        _json.dumps({"status": "ok", "generated_at": "2026-08-03T22:20:12Z"}))
    ts = LedgerBookSource(csv_path=str(csv)).upstream_changed_at()
    assert ts is not None and ts.tzinfo is not None
    assert ts.astimezone(timezone.utc).isoformat().startswith("2026-08-03T22:20:12")


def test_a_missing_sidecar_returns_none_rather_than_raising(tmp_path):
    """fetch() reports a missing sidecar properly. Raising here would turn a fetchable source into
    a silently skipped one — the exact failure this check exists to remove."""
    from kumo_strategies.runtime.executor.sources.ledger import LedgerBookSource

    csv = tmp_path / "ledger-book.csv"
    csv.write_text("symbol,open_date,close_date\n")
    assert LedgerBookSource(csv_path=str(csv)).upstream_changed_at() is None


# -- position claims vs broker reality ------------------------------------------------------------
# exec_position_state rows are CLAIMS made at buy-submit, not proof of a fill. In live, submit_order()
# schedules the venue call, so an order accepted locally can be rejected by Alpaca minutes later.
# The two failure directions are not symmetric and only one is dangerous.

def test_a_claim_with_no_broker_position_is_released_not_carried_forward():
    """A rejected buy leaves a claim behind. Harmless for selling — held_qty intersects with the
    account so it can never produce an order — but it consumes a position-cap slot and hides real
    capacity until something retires it."""
    import asyncio as aio

    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner

    released = []

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return {"GONE": 5, "REAL": 10}

        async def _drop_state(self, sym):
            released.append(sym)

    class Broker:
        def positions(self):
            return {"REAL": 10}          # GONE was claimed but never actually opened

        def equity(self):
            return 100_000.0

    class Jrn:
        rows = []

        async def write(self, kind, summary, **kw):
            Jrn.rows.append((kind, summary, kw.get("detail")))

        async def decided_this_session(self, session, slot=None):
            return True                  # stop after the reconcile; that is all this test covers

        async def explain(self, session, slot=None):
            return None                  # no recorded decision -> nothing to resume

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"REAL"}

        async def must_liquidate(self, sym):
            return False

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(), cfg=MomentumRotationConfig(),
               broker=Broker())
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert released == ["GONE"], f"stale claim not released: {released}"
    assert any("released 1 position claim" in s for _, s, _ in Jrn.rows), \
        "releasing a claim must leave a record, not happen silently"


def test_a_position_with_no_claim_is_the_dangerous_direction():
    """This is the case that must never be created: the strategy opened it, the claim is gone, so
    it reads as foreign, gets 'leaving them alone', and is never exited. Dropping the claim at
    SELL-submit produced exactly this whenever the venue rejected the sell — which is why the drop
    now happens at session start, against the broker, and not at submit."""
    import inspect

    from kumo_strategies.strategies.momentum_rotation import runner as pgrunner
    submit_src = inspect.getsource(pgrunner.PgSessionRunner._submit)
    sell_half = submit_src.split("if not st.may_submit_entries")[0]
    assert "_drop_state" not in sell_half, \
        "the sell path drops the claim again — a rejected sell will orphan a live position"


def test_a_decided_session_resumes_unsubmitted_orders_instead_of_doing_nothing():
    """The decision row is written BEFORE any order goes out, so a crash between the two leaves a
    session that has 'decided' and traded nothing. Bailing on 'already decided' means a give-back
    exit computed at 09:35 is never sent, ever."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    submitted = []

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return {"HELD": 5}

        async def _submit(self, session, enters, exits, held_qty, last, st, *, decision_slot=""):
            submitted.append({"enters": enters, "exits": exits})
            return len(enters) + len(exits)

    class Jrn:
        rows = []

        async def write(self, kind, summary, **kw):
            Jrn.rows.append((kind, summary))

        async def decided_this_session(self, session, slot=None):
            return True

        async def explain(self, session, slot=None):
            return {"detail": {"target_book": ["KEEP"], "reasons": {"HELD": "gave back 52%"}}}

        async def tail(self, n=50, kind=None, symbol=None):
            return []               # no order rows: nothing was ever submitted

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"HELD"}

        async def must_liquidate(self, sym):
            return False

    class Broker:
        def positions(self):
            return {"HELD": 5}

        def equity(self):
            return 100_000.0

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    res = aio.run(r.run(pd.DataFrame({"ticker": ["HELD"], "date": [pd.Timestamp("2026-08-03")],
                                      "close": [10.0]}), "2026-08-04"))
    assert submitted, "a decided-but-unsubmitted session did nothing on retry"
    assert "HELD" in submitted[0]["exits"], f"the recorded exit was not resumed: {submitted}"
    assert res.decided


def test_a_fully_submitted_session_does_not_resubmit():
    """Resume must be idempotent: an order row for a symbol means it was already attempted."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    submitted = []

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return {"HELD": 5}

        async def _submit(self, session, enters, exits, held_qty, last, st, *, decision_slot=""):
            submitted.append(1)
            return 0

    class Jrn:
        async def write(self, kind, summary, **kw):
            return 1

        async def decided_this_session(self, session, slot=None):
            return True

        async def explain(self, session, slot=None):
            return {"detail": {"target_book": [], "reasons": {"HELD": "gave back 52%"}}}

        async def tail(self, n=50, kind=None, symbol=None):
            # a RESULT row: the broker was actually called for this symbol
            return [{"session": "2026-08-04", "symbol": "HELD",
                     "detail": {"phase": "result", "ok": True}}]

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"HELD"}

        async def must_liquidate(self, sym):
            return False

    class Broker:
        def positions(self):
            return {"HELD": 5}

        def equity(self):
            return 100_000.0

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    res = aio.run(r.run(pd.DataFrame({"ticker": ["HELD"], "date": [pd.Timestamp("2026-08-03")],
                                      "close": [10.0]}), "2026-08-04"))
    assert submitted == [], "resubmitted an order that had already been attempted"
    assert res.blocked and "already decided" in res.blocked


def test_a_terminal_rejection_after_submit_time_ok_is_retried(monkeypatch):
    """#51: `phase="result", ok=true` means Nautilus accepted the order, not that the venue filled
    it. VCTR read exactly this at submit time, was rejected by Alpaca minutes later, and nothing ever
    retried it — resume treated the submit-time acceptance as done forever. A `phase="terminal",
    ok=false` row (the venue's real answer, written by `record_terminal`) must make it retryable
    again, distinct from a symbol that is still genuinely in flight with no terminal row at all."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    submitted = []

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return {"HELD": 5}

        async def _submit(self, session, enters, exits, held_qty, last, st, *, decision_slot=""):
            submitted.append(exits)
            return 0

    class Jrn:
        async def write(self, kind, summary, **kw):
            return 1

        async def decided_this_session(self, session, slot=None):
            return True

        async def explain(self, session, slot=None):
            return {"detail": {"target_book": [], "reasons": {"HELD": "gave back 52%"}}}

        async def tail(self, n=50, kind=None, symbol=None):
            return [
                {"session": "2026-08-04", "symbol": "HELD", "detail": {"phase": "result", "ok": True}},
                {"session": "2026-08-04", "symbol": "HELD",
                 "detail": {"phase": "terminal", "ok": False}},
            ]

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"HELD"}

        async def must_liquidate(self, sym):
            return False

    class Broker:
        def positions(self):
            return {"HELD": 5}

        def equity(self):
            return 100_000.0

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    res = aio.run(r.run(pd.DataFrame({"ticker": ["HELD"], "date": [pd.Timestamp("2026-08-03")],
                                      "close": [10.0]}), "2026-08-04"))
    assert submitted and "HELD" in submitted[0], (
        "a submit-time acceptance later rejected by the venue must be retried, not treated as done")
    assert res.decided


def test_submit_actually_varies_the_attempt_number_on_a_retried_symbol():
    """Not the property in isolation -- the WIRING. A prior terminal rejection this session must make
    `_submit` construct the retry's `OrderRequest` with `attempt >= 1`, or the real broker denies the
    resubmission as a duplicate of the order it already rejected, and #51's retry never reaches the
    venue a second time no matter how correct the retry DECISION is."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.runtime.executor.runner import RiskLimits
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    sent = []

    class Broker:
        def submit(self, req):
            from kumo_strategies.runtime.executor.broker import OrderResult
            sent.append(req)
            return OrderResult(True, "id", "detail", req)

        async def exit(self, req):
            return self.submit(req)

        def positions(self):
            return {}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 10.0

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def tail(self, n=400, kind=None, symbol=None):
            # VCTR was already terminally rejected once this session.
            return [{"session": "2026-08-14", "symbol": "VCTR",
                     "detail": {"phase": "terminal", "ok": False}}]

    r = PgSessionRunner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
                        cfg=MomentumRotationConfig(), broker=Broker(),
                        limits=RiskLimits(book_size=8))
    aio.run(r._submit("2026-08-14", (), ("VCTR",), {"VCTR": 88}, {"VCTR": 10.0}, State.TRADING))
    assert sent and sent[0].attempt == 1, (
        f"a symbol with one prior terminal rejection must retry at attempt=1, got {sent}")


def test_run_actually_threads_the_decision_slot_into_the_order_request():
    """Not the property in isolation -- the WIRING. `run()` knows which slot it decided under;
    `_submit` has to carry it into every `OrderRequest`, or MOMENTUM-002's open+5m attempt and a
    same-day open+130m retry hash to the SAME client_order_id and the retry is denied locally as a
    duplicate of the morning's order -- live, 2026-08-19 15:40 UTC, both FSM and VCTR."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.runtime.executor.runner import RiskLimits
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    sent = []

    class Broker:
        def submit(self, req):
            from kumo_strategies.runtime.executor.broker import OrderResult
            sent.append(req)
            return OrderResult(True, "id", "detail", req)

        async def exit(self, req):
            return self.submit(req)

        def positions(self):
            return {}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 10.0

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def tail(self, n=400, kind=None, symbol=None):
            return []

    r = PgSessionRunner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
                        cfg=MomentumRotationConfig(), broker=Broker(),
                        limits=RiskLimits(book_size=8))
    aio.run(r._submit("2026-08-19", (), ("FSM",), {"FSM": 933}, {"FSM": 10.0}, State.TRADING,
                      decision_slot="open+130m"))
    assert sent and sent[0].slot == "open+130m", (
        f"_submit did not carry the decision slot into the OrderRequest: {sent}")


def test_exits_are_routed_through_broker_exit_not_broker_submit():
    """FSM and VCTR were both refused `available: 0` while their own resting protective stop held
    the full quantity — the position was real, just reserved (issue 48,
    kumo-trading-platform issue 358). `broker.exit()` releases the reservation before sending; a plain
    `broker.submit()` on the exit path would hit the same wall every time. A fake broker with
    DIFFERENT outcomes for the two methods proves `_submit` actually calls `.exit()` for a SELL —
    not just that both happen to succeed the same way."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.runtime.executor.runner import RiskLimits
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    class Broker:
        def submit(self, req):
            raise AssertionError(f"exit routed through submit(), not exit(): {req}")

        async def exit(self, req):
            from kumo_strategies.runtime.executor.broker import OrderResult
            return OrderResult(True, "id", "released and sent", req)

        def positions(self):
            return {}

        def equity(self):
            return 100_000.0

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def tail(self, n=400, kind=None, symbol=None):
            return []

    r = PgSessionRunner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
                        cfg=MomentumRotationConfig(), broker=Broker(),
                        limits=RiskLimits(book_size=8))
    n = aio.run(r._submit("2026-08-19", (), ("FSM",), {"FSM": 933}, {}, State.TRADING))
    assert n == 1, "the exit through .exit() must still count as sent"


def test_the_exit_carries_the_RUNNERS_strategy_id_not_the_OrderRequest_default():
    """`release_for_exit` finds the protective stop BY strategy_id, so the wrong one releases nothing.

    `OrderRequest.strategy_id` defaults to the literal "MOMENTUM-002" (`broker.py:24`). One
    `PgSessionRunner` class serves three strategies — MOMENTUM-002, BCTROT-004 and QC345-003, the
    latter two moved to TRADING on 2026-08-19 — and cockpit's `SessionGateway` already passes the real
    id in (`momentum.py:206-213`). Dropping it on the way into the OrderRequest means BCTROT and QC345
    ask cockpit to release a stop belonging to a strategy that does not own the shares, find nothing to
    cancel, and have every exit refused — the same `available: 0` this whole fix removes, just arriving
    one layer further in.

    The runner here is deliberately NOT MOMENTUM-002: with the default id the assertion cannot fail,
    which is precisely how this slipped through.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.runtime.executor.runner import RiskLimits
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    seen = []

    class Broker:
        def submit(self, req):
            raise AssertionError(f"exit routed through submit(), not exit(): {req}")

        async def exit(self, req):
            from kumo_strategies.runtime.executor.broker import OrderResult
            seen.append(req.strategy_id)
            return OrderResult(True, "id", "released and sent", req)

        def positions(self):
            return {}

        def equity(self):
            return 100_000.0

    class Jrn:
        strategy_id = "BCTROT-004"

        async def write(self, *a, **k):
            return 1

        async def tail(self, n=400, kind=None, symbol=None):
            return []

    r = PgSessionRunner(pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
                        cfg=MomentumRotationConfig(), broker=Broker(),
                        limits=RiskLimits(book_size=8), strategy_id="BCTROT-004")
    aio.run(r._submit("2026-08-19", (), ("FSM",), {"FSM": 933}, {}, State.TRADING))

    assert seen == ["BCTROT-004"], (
        f"the exit went out as {seen} — the OrderRequest default, not the runner's own strategy_id, "
        "so release_for_exit would look for the wrong strategy's protective stop")


def test_resuming_one_slot_does_not_pick_up_a_different_slots_decision():
    """#54: `explain` used to ignore slot entirely and return whichever decision was most recent for
    the SESSION alone. A two-slot strategy (BCTROT-004: open+150m, close-20m) resuming its close-20m
    decision would silently get open+150m's book instead — the wrong exits, the wrong entries, and no
    error, because both rows share a session."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    resumed_with = []

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return {"OPEN_HELD": 5, "CLOSE_HELD": 5}

        async def _submit(self, session, enters, exits, held_qty, last, st, *, decision_slot=""):
            resumed_with.append(exits)
            return 0

    class Jrn:
        async def write(self, kind, summary, **kw):
            return 1

        async def decided_this_session(self, session, slot=None):
            return True

        async def explain(self, session, slot=None):
            # Two decisions, same session, different slots -- exactly BCTROT-004's shape.
            if slot == "close-20m":
                return {"detail": {"target_book": [], "reasons": {"CLOSE_HELD": "close-slot exit"}}}
            return {"detail": {"target_book": [], "reasons": {"OPEN_HELD": "open-slot exit"}}}

        async def tail(self, n=50, kind=None, symbol=None):
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"OPEN_HELD", "CLOSE_HELD"}

        async def must_liquidate(self, sym):
            return False

    class Broker:
        def positions(self):
            return {"OPEN_HELD": 5, "CLOSE_HELD": 5}

        def equity(self):
            return 100_000.0

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    res = aio.run(r.run(pd.DataFrame({"ticker": ["CLOSE_HELD"], "date": [pd.Timestamp("2026-08-03")],
                                      "close": [10.0]}), "2026-08-04", slot="close-20m"))
    assert resumed_with and "CLOSE_HELD" in resumed_with[0], (
        f"resuming slot=close-20m must use the close-20m decision, got {resumed_with}")
    assert not any("OPEN_HELD" in ex for ex in resumed_with), (
        "resuming one slot must not act on a different slot's decision")
    assert res.decided


def test_an_intent_row_alone_does_not_count_as_attempted():
    """The intent row is written BEFORE the broker call so a crash cannot leave an order with no
    audit trail. Treating it as proof of an attempt makes the crash-between-intent-and-submit case
    unrecoverable: the exit looks attempted forever and is never sent."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    submitted = []

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return {"HELD": 5}

        async def _submit(self, session, enters, exits, held_qty, last, st, *, decision_slot=""):
            submitted.append(exits)
            return len(exits)

    class Jrn:
        async def write(self, kind, summary, **kw):
            return 1

        async def decided_this_session(self, session, slot=None):
            return True

        async def explain(self, session, slot=None):
            return {"detail": {"target_book": [], "reasons": {"HELD": "gave back 52%"}}}

        async def tail(self, n=50, kind=None, symbol=None):
            # intent only — the process died before the broker was called
            return [{"session": "2026-08-04", "symbol": "HELD", "detail": {"phase": "intent"}}]

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"HELD"}

        async def must_liquidate(self, sym):
            return False

    class Broker:
        def positions(self):
            return {"HELD": 5}

        def equity(self):
            return 100_000.0

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    aio.run(r.run(pd.DataFrame({"ticker": ["HELD"], "date": [pd.Timestamp("2026-08-03")],
                                "close": [10.0]}), "2026-08-04"))
    assert submitted and "HELD" in submitted[0], "an unsent exit was treated as already attempted"


# -- _submit, driven for real -------------------------------------------------------------------
# Every earlier test stubbed _submit, so a name collision inside it -- the price getter bound to the
# same name as the position counter -- shipped and would have raised TypeError on the second entry,
# leaving a half-executed session. These drive the real method.

def _submit_fixture(**kw):
    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.runtime.executor.runner import RiskLimits
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    sent = []

    class Broker:
        def __init__(self):
            self.ok = kw.get("ok", True)

        def submit(self, req):
            from kumo_strategies.runtime.executor.broker import OrderResult
            sent.append(req)
            return OrderResult(self.ok, "id", "detail", req)

        async def exit(self, req):
            return self.submit(req)

        def positions(self):
            return kw.get("positions", {})

        def equity(self):
            #: OVERRIDABLE, and it matters. It was hardcoded to 100_000.0 — the same figure several
            #: tests pass as `allocated_equity` — so a mutant that IGNORED the allocation and used the
            #: account produced the identical quantity and those tests stayed green. The fixture made
            #: the two derivations agree. Pass `equity=` to make them differ.
            return kw.get("equity", 100_000.0)

        def last_price(self, sym, **_):
            return kw.get("prices", {}).get(sym)

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def tail(self, n=400, kind=None, symbol=None):
            return []

    class Runner(PgSessionRunner):
        async def _save_state(self, sym, st, qty=None):
            return None

        async def _drop_state(self, sym):
            return None

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=kw.get("cfg", MomentumRotationConfig()), broker=Broker(),
               limits=RiskLimits(**kw.get("limits", {})))
    return r, sent


# -- a configured exit rule this runner cannot honour (kumo-trading-platform issue 197 B12) ---------------------
# The rules are written twice, and only give_back_frac exists on the live side. Asking for one of the
# others used to be silently ignored: the book got entered on one ruleset and managed by a smaller
# one. The runner now refuses the ENTRY and keeps every exit path working — the opposite choice
# (raising) would also disable operator forced exits and LIQUIDATING, stranding real positions.

def _unsupported_cfg(monkeypatch=None):
    """A config the runner cannot honour.

    Since #197 P2 every ExitConfig field IS implemented live — both drivers call the same evaluator —
    so there is no longer a real example. The guard survives as a tripwire for the NEXT field someone
    adds, and that is what these tests exercise: shrink LIVE_SUPPORTED_EXITS to simulate a rule the
    evaluator does not yet handle.
    """
    from kumo_strategies.strategies.momentum_rotation import config as cfgmod
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig, MomentumRotationConfig)
    if monkeypatch is not None:
        monkeypatch.setattr(cfgmod, "LIVE_SUPPORTED_EXITS", frozenset({"give_back_frac"}))
    return MomentumRotationConfig(exits=ExitConfig(stall_days=10))


def test_an_exit_rule_live_cannot_honour_suppresses_entries(monkeypatch):
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(prices={"AAA": 10.0}, cfg=_unsupported_cfg(monkeypatch))
    n = aio.run(r._submit("2026-08-04", ("AAA",), (), {}, {"AAA": 10.0}, State.TRADING))
    assert sent == [], f"entered while an exit rule was unimplemented: {[s.symbol for s in sent]}"
    assert n == 0


def test_it_still_sells_while_entries_are_suppressed(monkeypatch):
    """The whole point of degrading rather than raising. A position we hold must still be closable."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(prices={"HHH": 10.0, "AAA": 10.0}, cfg=_unsupported_cfg(monkeypatch))
    aio.run(r._submit("2026-08-04", ("AAA",), ("HHH",), {"HHH": 25}, {"HHH": 10.0, "AAA": 10.0},
                      State.TRADING))
    assert [(s.symbol, s.side) for s in sent] == [("HHH", "SELL")], \
        f"the exit did not survive entry suppression: {[(s.symbol, s.side) for s in sent]}"


def test_a_suppressed_entry_is_terminal_for_resume(monkeypatch):
    """Codex review of PR #18 caught this: the guard originally wrote ONE summary risk row with no
    symbol. `_resume` rebuilds entries from the decision detail and treats a symbol as attempted only
    when a `phase=result` row names it — so every suppressed buy looked unsent and resume would
    submit exactly the orders the guard had refused. Worse, after a mid-session config fix it would
    submit them against a decision taken under the old config."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    rows = []

    class RecordingJrn:
        async def write(self, kind, msg, **k):
            rows.append({"kind": kind, "msg": msg, "symbol": k.get("symbol"),
                         "detail": k.get("detail") or {}})
            return 1

        async def tail(self, n=400, kind=None, symbol=None):
            return []

    r, sent = _submit_fixture(prices={"AAA": 10.0, "BBB": 10.0}, cfg=_unsupported_cfg(monkeypatch))
    r.journal = RecordingJrn()
    aio.run(r._submit("2026-08-04", ("AAA", "BBB"), (), {}, {"AAA": 10.0, "BBB": 10.0},
                      State.TRADING))

    marked = {x["symbol"] for x in rows if x["detail"].get("suppressed")}
    assert marked == {"AAA", "BBB"}, f"resume would replay the unmarked ones: {marked}"
    # Not a failed attempt — a failure is retried up to _MAX_SUBMIT_ATTEMPTS, and this is a decision
    # not to trade rather than a failure to.
    assert all(x["detail"].get("phase") == "result"
               for x in rows if x["detail"].get("suppressed"))


def test_a_suppressed_session_does_not_report_entries_it_never_made(monkeypatch):
    """The Nautilus layer journals `entered` straight off SessionResult. Returning the planned
    entries after the guard refused them would have the audit trail claim trades that never
    happened."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(prices={"AAA": 10.0}, cfg=_unsupported_cfg(monkeypatch))
    aio.run(r._submit("2026-08-04", ("AAA",), (), {}, {"AAA": 10.0}, State.TRADING))
    assert r._suppressed == ("AAA",)


def test_a_fully_supported_config_still_enters():
    """Guards the obvious regression: the check must not block the config we actually deploy."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig, MomentumRotationConfig)

    cfg = MomentumRotationConfig(exits=ExitConfig(give_back_frac=0.5))   # the live config
    r, sent = _submit_fixture(prices={"AAA": 10.0}, cfg=cfg)
    aio.run(r._submit("2026-08-04", ("AAA",), (), {}, {"AAA": 10.0}, State.TRADING))
    assert [s.symbol for s in sent] == ["AAA"]


def test_submit_places_every_entry_not_just_the_first():
    """The price getter was bound to `live`, the same name as the position counter, so `live += 1`
    after the first BUY added 1 to a function object."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(prices={"AAA": 10.0, "BBB": 20.0})
    n = aio.run(r._submit("2026-08-04", ("AAA", "BBB"), (), {}, {"AAA": 10.0, "BBB": 20.0},
                          State.TRADING))
    assert [s.symbol for s in sent] == ["AAA", "BBB"], f"only got {[s.symbol for s in sent]}"
    assert n == 2


def test_submit_prefers_the_live_price_over_the_stale_close():
    """Sizing a $10k budget off a $10 close when the name is $20 buys twice the intended notional."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(prices={"AAA": 20.0},
                              limits={"max_position_notional": 10_000.0, "book_size": 8})
    aio.run(r._submit("2026-08-04", ("AAA",), (), {}, {"AAA": 10.0}, State.TRADING))
    assert sent[0].qty == 500, f"sized off the close, not the live price: {sent[0].qty}"


def test_a_budget_too_small_to_buy_one_share_is_journalled_not_silent():
    """#334 (kumo-trading-platform): sizing off `allocated_equity` instead of the whole account brings the
    per-name budget within one order of magnitude of a real share price, where it used to be two.
    A name priced above the budget used to fall out of `enters` with a bare `continue` and no trace
    -- safe only while that branch was unreachable. It is reachable now."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.runtime.executor.runner import RiskLimits
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    written = []

    class Broker:
        def positions(self):
            return {}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return {"EXPENSIVE": 500.0}.get(sym)

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, kind, summary, **kw):
            written.append((kind, summary, kw))
            return 1

        async def tail(self, n=400, kind=None, symbol=None):
            return []

    class Runner(PgSessionRunner):
        async def _save_state(self, sym, st, qty=None):
            return None

        async def _drop_state(self, sym):
            return None

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker(),
               limits=RiskLimits(allocated_equity=2_000.0, book_size=8))
    n = aio.run(r._submit("2026-08-04", ("EXPENSIVE",), (), {}, {"EXPENSIVE": 500.0}, State.TRADING))
    assert n == 0, "a budget too small for one share must not count as a submission"
    risk_rows = [w for w in written if w[0] == "risk" and "EXPENSIVE" in w[1]]
    assert risk_rows, (
        f"a symbol dropped for an unaffordable budget must be journalled, not silently skipped: "
        f"{written}")


def test_a_ZERO_allocation_sizes_off_NOTHING_not_the_account():
    """`allocated_equity=0.0` is falsy, and the sizing line was `alloc or broker.equity()`.

    A lane cockpit granted nothing therefore sized every entry off the ACCOUNT — the one-position
    failure the comment above that line exists to prevent, arriving through the VALUE rather than
    through a missing one. kumo-trading-platform measured the live exposure on kumo-staging (2026-08-24):
    BCTROT-004 allocated 100000, MANUAL-001 / MOMENTUM-002 / QC345-003 / TECHIVOL-005 all allocated
    0, account equity 999,215.43. Three unrelated accidents were the only thing between a
    zero-allocation lane and ~1M of BCTROT's capital — two disabled by flag, one with no lifecycle
    row, and an `exec_action_log` still empty on that stack.

    Bound to `_submit` on the REAL runner rather than to a copy of the expression: the sibling
    `test_allocated_equity.py` keeps a second copy of this formula and its own docstring records that
    the copy drifted from production once already. This one cannot.

    `None` is untouched and `test_a_budget_too_small_to_buy_one_share_is_journalled_not_silent`
    covers a real allocation, so the fallback is narrowed, not deleted."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(prices={"AAA": 20.0},
                              limits={"allocated_equity": 0.0, "book_size": 8})
    n = aio.run(r._submit("2026-08-04", ("AAA",), (), {}, {"AAA": 20.0}, State.TRADING))
    assert sent == [], (
        f"a lane allocated 0.0 sized off the 100k account and submitted "
        f"{[(o.symbol, o.qty) for o in sent]}")
    assert n == 0, "a zero allocation must degrade to inaction, not to a submission"


def test_a_NONZERO_allocation_sizes_EXACTLY_as_it_did_before_the_zero_fix():
    """be244d9 changed the 0.0 and `None` branches ONLY. Every funded lane must be untouched.

    Asked by kumo-trading-platform 2026-08-24 before putting BCTROT-004 on ibkr-paper-retired: that lane is allocated
    100000, has NEVER run a session (`exec_action_log` on kumo-staging is empty), and so has no dryrun
    history to compare a new pin against. The question is whether the pin bump moves its sizing.

    It does not. The old expression was `allocated_equity or <account>`; 100000 is truthy, so it
    returned 100000, and the new `float(alloc)` returns 100000.0 — equal, and `slot` is a float in
    both. Verified across the inputs that matter rather than argued:

        alloc=100000.0  old=100000.0   new=100000.0   identical
        alloc=100000    old=100000     new=100000.0   identical (int/float compare equal, same qty)
        alloc=20000.0   old=20000.0    new=20000.0    identical
        alloc=None      old=<account>  new=<account>  identical
        alloc=0.0       old=<account>  new=0.0        DIFFERENT  <- the entire point of be244d9

    Pinned to the real numbers for BCTROT's live configuration: 100000 * 0.80 / 8 = 10000 a name,
    which is under `max_position_notional` (20000), so the cap does not bind and `allocated_equity` is
    the only thing sizing that lane.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State
    from kumo_strategies.runtime.executor.runner import RiskLimits

    # THE ACCOUNT MUST DIFFER FROM THE ALLOCATION or this test cannot fail. The fixture's account
    # was 100_000.0, the same number as BCTROT's allocation, so a mutant that ignored the allocation
    # entirely and sized off the account returned the SAME 100 shares and this passed. Caught by
    # biting it. 1_000_000.00 is kumo-staging's actual reported account equity.
    r, sent = _submit_fixture(prices={"AAA": 100.0}, equity=1_000_000.00,
                              limits={"allocated_equity": 100_000.0, "book_size": 8})
    aio.run(r._submit("2026-08-24", ("AAA",), (), {}, {"AAA": 100.0}, State.TRADING))

    assert [(o.symbol, o.qty) for o in sent] == [("AAA", 100)], (
        f"a funded lane's sizing moved: {[(o.symbol, o.qty) for o in sent]} — 100000 * 0.80 / 8 = "
        f"10000 at 100.0 is 100 shares, and be244d9 was supposed to touch only 0.0 and None")
    assert sent[0].qty * 100.0 < RiskLimits().max_position_notional, (
        "the notional cap now binds for this configuration, so this test is measuring the cap rather "
        "than the allocation and no longer answers the question it was written for")


def test_an_accepted_sell_does_not_free_a_slot_for_a_new_buy():
    """`ok` means accepted, not filled. Counting it as closed lets buys fill slots the unfilled
    sells have not vacated, and the book ends over the position cap."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    # THE CAP, not the book size. The operator's decision 2026-08-21: a book at `book_size` must be able to
    # rotate, so a sold name no longer blocks its replacement — that is the whole point of the split.
    # What must still hold is the HARD CEILING: at `max_positions`, an accepted-but-unfilled sell
    # does not create room, because `ok` means accepted and not filled.
    held = {f"H{i}": 10 for i in range(12)}         # at the CAP of 12, not merely the book of 8
    r, sent = _submit_fixture(prices={"NEW": 10.0}, limits={"book_size": 8, "max_positions": 12})
    aio.run(r._submit("2026-08-04", ("NEW",), ("H0",), held, {"H0": 10.0, "NEW": 10.0},
                      State.TRADING))
    assert [s.symbol for s in sent] == ["H0"], \
        f"a buy took a slot the sell had not actually vacated: {[s.symbol for s in sent]}"


# -- the ownership boundary ----------------------------------------------------------------------
# This survived three reviews in different forms. A claim used to mean "this strategy owns the
# symbol", so the whole account quantity was fair game; then a fix capped by broker attribution but
# fell back to the account quantity when attribution was missing, which is the same bug on the path
# that matters most. The claim's QUANTITY is the cap that holds in every case.

def _owner_fixture(account, claims, attributed=None):
    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    seen = {}

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return dict(claims)

        async def _drop_state(self, sym):
            return None

        async def _save_state(self, sym, st, qty=None):
            return None

        async def _resume(self, session, st, held_qty, panel, slot=None):
            # held_qty is the computed ownership boundary — capture it where it is handed on
            seen["held_qty"] = dict(held_qty)
            from kumo_strategies.runtime.executor.runner import SessionResult
            return SessionResult(session, st.value, False, blocked="captured")

    class Broker:
        def positions(self):
            return dict(account)

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 10.0

    if attributed is not None:
        Broker.strategy_positions = lambda self: dict(attributed)

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def decided_this_session(self, session, slot=None):
            return True

        async def explain(self, session, slot=None):
            return None

        async def tail(self, n=50, kind=None, symbol=None):
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return set(account)

        async def must_liquidate(self, sym):
            return False

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    return r, seen


def test_an_exit_never_reaches_past_what_this_strategy_claimed():
    """MOMENTUM holds 10 AAPL, a manual trade holds 90. The account reports 100. Selling 100 to
    close our 10 liquidates someone else's position."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAPL": 100}, claims={"AAPL": 10},
                             attributed={"AAPL": 10})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"AAPL": 10}, f"would have sold {seen['held_qty']}"


def test_missing_attribution_still_caps_at_the_claim_not_the_account():
    """A broker that cannot attribute per strategy must not widen the exit to the whole account —
    and must not be unable to exit at all either."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAPL": 100}, claims={"AAPL": 10}, attributed=None)
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"AAPL": 10}


def test_a_claim_with_no_quantity_sells_nothing():
    """Legacy rows predate the qty column. Zero must mean no order, not 'unknown, take the lot'."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAPL": 100}, claims={"AAPL": 0}, attributed={"AAPL": 100})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {}


def test_a_broker_position_we_never_recorded_is_adopted_not_orphaned():
    """A buy that filled while the process was dying leaves a real position with no claim. Unclaimed
    reads as foreign, and foreign is never exited by anything — including liquidation."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAPL": 10}, claims={}, attributed={"AAPL": 10})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"AAPL": 10}, "the orphaned position was left unmanaged"


def test_give_back_prices_on_the_current_market_not_yesterdays_close():
    """The session fires at the open, so the last daily bar is yesterday. A position that gapped
    through the give-back level overnight is precisely the one that must exit — entry 10, peak 20,
    yesterday 18, opening at 12 breaches the rule, and reading 18 holds it."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig, MomentumRotationConfig)

    from kumo_strategies.strategies.momentum_rotation.exits import LIVE, TrailState

    class Runner(PgSessionRunner):
        async def _load_state(self, symbols):
            # entry 10, peak 20, observed — so the give-back rule is genuinely armed
            return {"AAA": TrailState(entry_px=10.0, peak_px=20.0, quality=LIVE)}

        async def _save_state(self, sym, st, qty=None):
            return None

    class Broker:
        def positions(self):
            return {"AAA": 10}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 12.0                            # gapped down overnight

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(exits=ExitConfig(give_back_frac=0.5)), broker=Broker())
    exits = aio.run(r._trail_exits({"AAA": 10}, {"AAA": 18.0}))       # stale close says hold
    assert "AAA" in exits, f"gap through the give-back level did not fire: {exits}"


def test_liquidating_flattens_even_when_the_session_already_decided():
    """An operator flipping to LIQUIDATING mid-morning used to be honoured only by the NEXT
    session, and if today had already decided, the resume path replayed the old decision instead.
    The UI said liquidating; the account stayed long."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    sent = []

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return {"AAA": 10, "BBB": 5}

        async def _submit(self, session, enters, exits, held_qty, last, st, *, decision_slot=""):
            sent.append({"enters": enters, "exits": exits})
            return len(exits)

    class Broker:
        def positions(self):
            return {"AAA": 10, "BBB": 5}

        def equity(self):
            return 100_000.0

        def strategy_positions(self):
            return {"AAA": 10, "BBB": 5}

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def decided_this_session(self, session, slot=None):
            return True                      # already decided today

        async def explain(self, session, slot=None):
            return {"detail": {"target_book": ["AAA", "BBB"], "reasons": {}}}

        async def tail(self, n=50, kind=None, symbol=None):
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"AAA", "BBB"}

        async def must_liquidate(self, sym):
            return False

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.LIQUIDATING, "operator"),
               cfg=MomentumRotationConfig(), broker=Broker())
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert sent, "LIQUIDATING did nothing on an already-decided session"
    assert set(sent[0]["exits"]) == {"AAA", "BBB"}
    assert sent[0]["enters"] == ()


def test_a_blacklist_after_the_decision_still_exits_that_symbol():
    """Blacklisting a held name is the operator saying they know something the ranking does not."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    sent = []

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return {"AAA": 10, "BBB": 5}

        async def _submit(self, session, enters, exits, held_qty, last, st, *, decision_slot=""):
            sent.append(exits)
            return len(exits)

    class Broker:
        def positions(self):
            return {"AAA": 10, "BBB": 5}

        def equity(self):
            return 100_000.0

        def strategy_positions(self):
            return {"AAA": 10, "BBB": 5}

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def decided_this_session(self, session, slot=None):
            return True

        async def explain(self, session, slot=None):
            return {"detail": {"target_book": ["AAA", "BBB"], "reasons": {}}}

        async def tail(self, n=50, kind=None, symbol=None):
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"AAA", "BBB"}

        async def must_liquidate(self, sym):
            return sym == "AAA"

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert sent and sent[0] == ("AAA",), f"blacklisted symbol not exited: {sent}"


def test_an_entry_is_skipped_rather_than_sized_off_a_stale_close():
    """Yesterday's close is wrong by exactly the overnight gap. A skipped entry costs an
    opportunity once; an entry sized on a stale price costs money every time."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(prices={})          # no live price for anything
    aio.run(r._submit("2026-08-04", ("AAA",), (), {}, {"AAA": 10.0}, State.TRADING))
    assert sent == [], "sized an entry off a stale close"


def test_a_position_the_broker_cannot_attribute_is_still_sellable():
    """With the durable cache off, a position reconciled in after a restart comes back without a
    Nautilus strategy id. Reading that as 'we own none of it' made give-back and even LIQUIDATING
    refuse to sell a position we opened — the mirror of the bug attribution was added to fix, and
    just as unexitable. The CLAIM is what caps us; attribution only narrows where it reports."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAA": 10}, claims={"AAA": 10}, attributed={})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"AAA": 10}, f"unexitable after restart: {seen['held_qty']}"


def test_attribution_still_narrows_when_it_reports_the_symbol():
    """The original bug must stay fixed: 10 ours beside a manual 90 sells 10, not 100."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAA": 100}, claims={"AAA": 100}, attributed={"AAA": 10})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"AAA": 10}


def test_a_stale_export_is_refused(tmp_path):
    """The staleness rule itself, stated explicitly instead of ridden on by accident.

    Three fixtures below used to hardcode `generated_at`, so they drifted past MAX_META_AGE_HOURS and
    began failing on age while testing checksums and row counts. Giving them a fresh timestamp fixed
    the rot but removed the only coverage the age check had. This test owns it: it states its own age
    rather than depending on what today's date happens to be, so it can neither rot nor be the
    incidental cause of someone else's failure.

    The behaviour matters — a stale book must not drive a session. On 6 Aug the pool was a day behind
    and every exit decision rested on it.
    """
    import hashlib
    import json as _json
    from datetime import timedelta

    from kumo_strategies.runtime.executor.sources.base import SourceError
    from kumo_strategies.runtime.executor.sources.ledger import (
        MAX_META_AGE_HOURS, LedgerBookSource)

    csv = tmp_path / "ledger-book.csv"
    csv.write_text("symbol,open_date,close_date\nAAA,2026-08-01,\n")
    body = csv.read_bytes()
    old = datetime.now(timezone.utc) - timedelta(hours=MAX_META_AGE_HOURS + 1)
    (tmp_path / "ledger-book.meta.json").write_text(_json.dumps({
        "status": "ok", "generated_at": old.strftime("%Y-%m-%dT%H:%M:%SZ"), "row_count": 1,
        "csv_sha256": hashlib.sha256(body).hexdigest()}))

    with pytest.raises(SourceError) as e:
        LedgerBookSource(csv_path=str(csv)).fetch()
    assert "old" in str(e.value).lower() or "stale" in str(e.value).lower()


def test_a_fresh_export_is_accepted(tmp_path):
    """The other half — otherwise the test above passes just as well against a source that refuses
    everything."""
    import hashlib
    import json as _json

    from kumo_strategies.runtime.executor.sources.ledger import LedgerBookSource

    csv = tmp_path / "ledger-book.csv"
    csv.write_text("symbol,open_date,close_date\nAAA,2026-08-01,\n")
    body = csv.read_bytes()
    (tmp_path / "ledger-book.meta.json").write_text(_json.dumps({
        "status": "ok", "generated_at": _fresh_generated_at(), "row_count": 1,
        "csv_sha256": hashlib.sha256(body).hexdigest()}))

    assert LedgerBookSource(csv_path=str(csv)).fetch() is not None


def test_a_csv_that_does_not_match_its_sidecar_is_refused(tmp_path):
    """The CSV and sidecar are separate atomic writes. Reading between them pairs a new book with
    the old manifest, and status/generated_at/schema all validate the wrong file. Same shape as the
    pool sitting four hours behind ledger-tool's fix: each layer internally consistent, pairing wrong."""
    import hashlib
    import json as _json

    import pytest

    from kumo_strategies.runtime.executor.sources.base import SourceError
    from kumo_strategies.runtime.executor.sources.ledger import LedgerBookSource

    csv = tmp_path / "ledger-book.csv"
    csv.write_text("symbol,open_date,close_date\nAAA,2026-08-01,\n")
    meta = {"status": "ok", "generated_at": _fresh_generated_at(), "row_count": 1,
            "csv_sha256": hashlib.sha256(b"a different export entirely").hexdigest()}
    (tmp_path / "ledger-book.meta.json").write_text(_json.dumps(meta))
    with pytest.raises(SourceError) as e:
        LedgerBookSource(csv_path=str(csv)).fetch()
    assert "sidecar" in str(e.value)


def test_a_row_count_mismatch_is_refused(tmp_path):
    import json as _json

    import pytest

    from kumo_strategies.runtime.executor.sources.base import SourceError
    from kumo_strategies.runtime.executor.sources.ledger import LedgerBookSource

    csv = tmp_path / "ledger-book.csv"
    csv.write_text("symbol,open_date,close_date\nAAA,2026-08-01,\n")
    (tmp_path / "ledger-book.meta.json").write_text(
        _json.dumps({"status": "ok", "generated_at": _fresh_generated_at(), "row_count": 975}))
    with pytest.raises(SourceError) as e:
        LedgerBookSource(csv_path=str(csv)).fetch()
    assert "manifest" in str(e.value)


def test_a_recently_confirmed_lot_survives_expiry(tmp_path):
    """The cohort expiry could never keep: long holds he stops posting about. AEM, CGAU, SCCO and
    WPM are held for many months and confirmed by full reviews, but an age rule retires exactly
    those. A confirmation at the corpus edge outranks age."""
    import hashlib
    import json as _json

    from kumo_strategies.runtime.executor.sources.ledger import LedgerBookSource

    csv = tmp_path / "ledger-book.csv"
    csv.write_text(
        "symbol,open_date,close_date,last_confirmed_held\n"
        "OLD,2025-01-01,,2026-07-25\n"        # 580d old, confirmed at the edge -> keep
        "STALE,2025-01-01,,2025-03-01\n"      # 580d old, last confirmed long ago -> expire
        "NEVER,2025-01-01,,\n"                # never confirmable -> expiry decides -> expire
        "RECENT,2026-07-20,,\n"               # young, no confirmation -> expiry keeps it
    )
    body = csv.read_bytes()
    (tmp_path / "ledger-book.meta.json").write_text(_json.dumps({
        "status": "ok", "generated_at": _fresh_generated_at(), "row_count": 4,
        "csv_sha256": hashlib.sha256(body).hexdigest(),
        "latest_full_review": "2026-07-25"}))

    res = LedgerBookSource(csv_path=str(csv)).fetch(
        asof=__import__("datetime").datetime(2026, 8, 4, tzinfo=__import__("datetime").timezone.utc))
    got = set(res.symbols)
    assert "OLD" in got, "a lot confirmed at the corpus edge was retired on age"
    assert "RECENT" in got
    assert "STALE" not in got, "a lot last confirmed months ago should fall back to expiry"
    assert "NEVER" not in got, "an unconfirmable lot past expiry should still expire"


def test_a_failed_source_retries_quickly_rather_than_waiting_out_its_cadence():
    """A failed refresh still stamps last_run, so the cadence clock restarts on failure. With a
    30-minute cadence a transient fault blocks the source for 30 minutes — and the runner treats a
    stale source as a HARD BLOCK on deciding, so it can cost the whole session.

    That is not hypothetical: a missing container mount failed the refresh at 13:05, the cadence
    pushed the retry to 13:35, and the session fires at 13:35."""
    from datetime import datetime, timezone

    from kumo_strategies.runtime.executor.pgjobs import _due

    now = datetime(2026, 8, 4, 13, 20, tzinfo=timezone.utc)
    failed = _spec_row(enabled=True, every_minutes=30, last_status="failed",
                       last_run=datetime(2026, 8, 4, 13, 5, tzinfo=timezone.utc))
    assert _due(failed, now=now) is True, "a failed source waited out its full cadence"

    # ...but not instantly, or a hard-down upstream is hammered every tick
    just_failed = _spec_row(enabled=True, every_minutes=30, last_status="failed",
                            last_run=datetime(2026, 8, 4, 13, 19, 30, tzinfo=timezone.utc))
    assert _due(just_failed, now=now) is False

    # a SUCCESSFUL source keeps its normal cadence
    ok = _spec_row(enabled=True, every_minutes=30, last_status="ok",
                   last_run=datetime(2026, 8, 4, 13, 5, tzinfo=timezone.utc))
    assert _due(ok, now=now) is False


def test_resume_retries_a_failed_submit_but_not_forever():
    """Counting any result row as an attempt — including failures — meant a session that failed for
    an ENVIRONMENTAL reason could never recover. The first live run refused all 8 entries because
    instrument definitions were missing; once that was fixed, resume still said "already decided"
    because those failures looked like attempts."""
    from kumo_strategies.strategies.momentum_rotation.runner import _MAX_SUBMIT_ATTEMPTS

    def attempted(rows):
        results = {}
        for r in rows:
            d = r.get("detail") or {}
            if r["session"] == "S" and r.get("symbol") and d.get("phase") == "result":
                results.setdefault(r["symbol"], []).append(bool(d.get("ok")))
        return {s for s, oks in results.items() if any(oks) or len(oks) >= _MAX_SUBMIT_ATTEMPTS}

    ok = [{"session": "S", "symbol": "A", "detail": {"phase": "result", "ok": True}}]
    assert attempted(ok) == {"A"}, "a successful submit must not be repeated"

    one_fail = [{"session": "S", "symbol": "B", "detail": {"phase": "result", "ok": False}}]
    assert attempted(one_fail) == set(), "a single failure must stay retryable"

    many = [{"session": "S", "symbol": "C", "detail": {"phase": "result", "ok": False}}] * _MAX_SUBMIT_ATTEMPTS
    assert attempted(many) == {"C"}, "a repeatedly failing symbol must stop being retried"


def test_an_adopted_position_uses_the_brokers_real_entry_not_todays_quote():
    """#197 B1. Seeding entry from the current quote is how the trail became fiction. Nautilus
    already carries avg_px_open on every position; the runner simply never asked."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig, MomentumRotationConfig)
    from kumo_strategies.strategies.momentum_rotation.exits import ADOPTED

    saved = {}

    class Runner(PgSessionRunner):
        async def _load_state(self, symbols):
            return {}                                  # nothing on record: must adopt

        async def _save_state(self, sym, st, qty=None):
            saved[sym] = st

    class Broker:
        def positions(self):
            return {"AAA": 10}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 12.0                                # today's quote, far from the real fill

        def position_entries(self):
            return {"AAA": 47.5}                       # what it actually opened at

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(exits=ExitConfig(give_back_frac=0.5)), broker=Broker())
    aio.run(r._trail_exits({"AAA": 10}, {}))
    trail = r._pending_trail["AAA"][0]
    assert trail.entry_px == 47.5, f"seeded from the quote again: {trail}"
    assert trail.quality == ADOPTED, "peak is still unknown — quality must say so"
    assert saved == {}, "the trail must not be persisted before the decision row is won"


def test_adoption_falls_back_to_the_quote_when_the_broker_cannot_attribute():
    """A broker with no per-strategy attribution (a plain REST account view) cannot know the entry.
    Falling back is correct; pretending otherwise is not."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig, MomentumRotationConfig)

    saved = {}

    class Runner(PgSessionRunner):
        async def _load_state(self, symbols):
            return {}

        async def _save_state(self, sym, st, qty=None):
            saved[sym] = st

    class Broker:                                      # no position_entries at all
        def positions(self):
            return {"AAA": 10}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 12.0

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(exits=ExitConfig(give_back_frac=0.5)), broker=Broker())
    aio.run(r._trail_exits({"AAA": 10}, {}))
    assert r._pending_trail["AAA"][0].entry_px == 12.0


def test_the_orphan_sweep_adopts_at_the_brokers_entry_too():
    """Codex caught this on review: the orphan sweep in run() seeded entry from the quote on its own,
    and because it PERSISTS a row, the later _trail_exits found state and never re-adopted. Fixing
    the evaluator path alone left the live path exactly as wrong as before."""
    import asyncio as aio

    saved = {}
    r, _seen = _owner_fixture(account={"AAPL": 10}, claims={}, attributed={"AAPL": 10})

    async def capture(sym, st, qty=None):
        saved[sym] = st
    r._save_state = capture
    type(r.broker).position_entries = lambda self: {"AAPL": 47.5}    # real fill, quote says 10.0

    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert saved["AAPL"].entry_px == 47.5, f"orphan seeded from the quote: {saved.get('AAPL')}"


def test_a_session_blocked_after_the_exits_ran_does_not_advance_the_trail():
    """The loser of a concurrent race, and any session stopped by a data gate, must leave the trail
    alone: it submitted nothing, so counting it against max_hold_days would age positions out of the
    book on days we refused to act."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig, MomentumRotationConfig)
    from kumo_strategies.strategies.momentum_rotation.exits import LIVE, TrailState

    saved = {}

    class Runner(PgSessionRunner):
        async def _load_state(self, symbols):
            return {"AAA": TrailState(entry_px=10.0, peak_px=20.0, quality=LIVE,
                                      sessions_held=3)}

        async def _save_state(self, sym, st, qty=None):
            saved[sym] = st

    class Broker:
        def positions(self):
            return {"AAA": 10}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 19.0

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(exits=ExitConfig(give_back_frac=0.5)), broker=Broker())
    aio.run(r._trail_exits({"AAA": 10}, {}))

    assert saved == {}, "state was persisted by a session that had not yet won its decision row"
    assert r._pending_trail["AAA"][0].sessions_held == 4, "the advance itself must still be computed"


# -- inverse-vol sizing reaching the LIVE runner (#26) -------------------------------------------
# `max_correlation` and `inverse_vol_sizing` were inert in production: this runner called decide()
# without `corr`/`vol` and then sized every entry at a flat
# `equity * max_deployed_frac / max_positions`, ignoring `dec.weights`. Either omission alone makes
# the flag a no-op that typechecks, deploys and reports no error.
#
# Both halves matter and the second is the one that was missed in the backtest for weeks: passing
# `vol` while still sizing flat leaves the flag just as dead. So these test the SIZE of the order,
# not whether an input was threaded.

def _ivol_cfg():
    from kumo_strategies.strategies.momentum_rotation.config import (
        MomentumRotationConfig, PortfolioConfig)
    return MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=4, inverse_vol_sizing=True))


def test_live_entries_size_from_the_decision_weights():
    """A low-vol name must get more than one equal-weight slot of dollars.

    Book of four, `AAA` weighted 0.40 against an equal 0.25 — so 1.6 slots. At
    equity 100k, max_deployed_frac 0.75 and max_positions 4, a slot is 18,750;
    1.6 slots at $10 is 3,000 shares against the flat 1,875.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(cfg=_ivol_cfg(), prices={"AAA": 10.0},
                              limits={"book_size": 4, "max_deployed_frac": 0.75,
                                      "max_position_notional": 1_000_000.0})
    weights = {"AAA": 0.40, "BBB": 0.20, "CCC": 0.20, "DDD": 0.20}
    aio.run(r._submit("2026-08-04", ("AAA",), (), {"BBB": 1, "CCC": 1, "DDD": 1},
                      {"AAA": 10.0}, State.TRADING, weights=weights))
    assert sent[0].qty == 3000, (
        f"live sizing ignored dec.weights: got {sent[0].qty}, expected 3000. A flat 1875 means the "
        "weights are still being computed and discarded, which is what made inverse_vol_sizing "
        "inert in production")


def test_equal_weights_size_exactly_as_the_flat_budget_did():
    """The no-change guarantee. Every config running today is equal-weighted, and must be untouched.

    Equal weight makes `w = 1/len(book)`, so the multiple is exactly 1.0 and the arithmetic reduces
    to the previous formula rather than merely approximating it.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(cfg=_ivol_cfg(), prices={"AAA": 10.0},
                              limits={"book_size": 4, "max_deployed_frac": 0.75,
                                      "max_position_notional": 1_000_000.0})
    weights = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}
    aio.run(r._submit("2026-08-04", ("AAA",), (), {"BBB": 1, "CCC": 1, "DDD": 1},
                      {"AAA": 10.0}, State.TRADING, weights=weights))
    assert sent[0].qty == 1875, f"equal weight must reduce to the flat slot, got {sent[0].qty}"


def test_max_position_notional_still_binds_after_the_weight_is_applied():
    """A hard risk limit is not negotiable by a sizing rule.

    Without ordering the cap AFTER the weight, a low-vol name earning three slots would breach the
    per-position notional ceiling — the sizing rule would have quietly raised a risk limit.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(cfg=_ivol_cfg(), prices={"AAA": 10.0},
                              limits={"book_size": 4, "max_deployed_frac": 0.75,
                                      "max_position_notional": 20_000.0})
    weights = {"AAA": 0.70, "BBB": 0.10, "CCC": 0.10, "DDD": 0.10}
    aio.run(r._submit("2026-08-04", ("AAA",), (), {"BBB": 1, "CCC": 1, "DDD": 1},
                      {"AAA": 10.0}, State.TRADING, weights=weights))
    assert sent[0].qty == 2000, (
        f"max_position_notional breached: {sent[0].qty} shares at $10 exceeds the $20k cap")


def test_the_runner_honours_weights_whenever_the_engine_supplies_them():
    """The runner must not decide whether the engine's output matters.

    This test previously asserted the OPPOSITE — that a config without `inverse_vol_sizing` had to
    size flat even when weights were supplied — and called it defence in depth. That was wrong, and
    it encoded the #26 defect as a requirement: a runner second-guessing the config it was handed is
    exactly how `max_correlation` and `inverse_vol_sizing` stayed inert for months.

    The correct separation is that the FLAG controls what `_weights()` returns, and the runner
    consumes whatever it gets. Under equal weight the engine returns `1/len(book)`, so the multiple
    is exactly 1.0 and the flat budget falls out by arithmetic rather than by a conditional. Any
    future weighting scheme is then honoured automatically instead of being silently ignored until
    someone remembers to flip an unrelated flag.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State
    from kumo_strategies.strategies.momentum_rotation.config import (
        MomentumRotationConfig, PortfolioConfig)

    plain = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=4))
    limits = {"book_size": 4, "max_deployed_frac": 0.75,
              "max_position_notional": 1_000_000.0}

    # Equal weights, flag off: must reduce exactly to the flat slot.
    r, sent = _submit_fixture(cfg=plain, prices={"AAA": 10.0}, limits=limits)
    aio.run(r._submit("2026-08-04", ("AAA",), (), {"BBB": 1, "CCC": 1, "DDD": 1},
                      {"AAA": 10.0}, State.TRADING,
                      weights={n: 0.25 for n in ("AAA", "BBB", "CCC", "DDD")}))
    assert sent[0].qty == 1875, f"equal weight must give the flat slot, got {sent[0].qty}"

    # Non-equal weights, same flag-off config: must be honoured, not discarded.
    r2, sent2 = _submit_fixture(cfg=plain, prices={"AAA": 10.0}, limits=limits)
    aio.run(r2._submit("2026-08-04", ("AAA",), (), {"BBB": 1, "CCC": 1, "DDD": 1},
                       {"AAA": 10.0}, State.TRADING,
                       weights={"AAA": 0.40, "BBB": 0.20, "CCC": 0.20, "DDD": 0.20}))
    assert sent2[0].qty == 3000, (
        f"the runner discarded supplied weights because a flag was off: got {sent2[0].qty}. That is "
        "the runner overriding the engine, which is the defect #26 documents.")


# -- a book at its size must be able to ROTATE ------------------------------------------------------
def test_a_book_at_its_BOOK_SIZE_can_rotate():
    """MOMENTUM-002 COULD NOT COMPLETE A ROTATION BETWEEN 2026-08-06 AND 2026-08-21.

    One number was doing two jobs. `max_positions` was both the hard cap and the sizing divisor
    (`slot = equity * max_deployed_frac / max_positions`). MOMENTUM's researched book is n_hold=8 and
    its limits are a bare `RiskLimits()`, so the cap was ALSO 8 — book size equal to the ceiling,
    zero rotation headroom. It sold N and had all N replacements refused by the names it had just
    sold, because the cap counts what is held right now and deliberately does not deduct sells
    submitted moments ago.

    Position-cap refusals per session, from the live journal:

        2026-08-06  6    2026-08-10  2    2026-08-11  1    2026-08-13  1
        2026-08-14  1    2026-08-19  6    2026-08-21  2   <- sold WHD and XLV, entered nothing, 8 -> 6

    19 entries lost across seven sessions. Nothing errored: each refusal journalled a tidy
    "position cap" while the book shrank and the strategy looked healthy.

    RESOLVED BY SPLITTING THE NUMBER (the operator's decision, 2026-08-21, option (b) of two). `book_size` is
    the researched book and drives sizing; `max_positions` is the hard ceiling and must exceed it.
    The alternative — deducting submitted exits from the live count — was one line and reversed a
    deliberate guard, so it was written, seen to turn that guard red, and reverted.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    held = {f"H{i}": 10 for i in range(8)}                      # a FULL book, at book_size
    r, sent = _submit_fixture(prices={"NEW1": 10.0, "NEW2": 10.0},
                              limits={"book_size": 8, "max_positions": 12})
    aio.run(r._submit("2026-08-21", ("NEW1", "NEW2"), ("H0", "H1"), held,
                      {"H0": 10.0, "H1": 10.0, "NEW1": 10.0, "NEW2": 10.0}, State.TRADING))
    buys = sorted(s.symbol for s in sent if s.side == "BUY")
    assert buys == ["NEW1", "NEW2"], f"a full book still cannot rotate: bought {buys}"


# -- the ownership boundary ----------------------------------------------------------------------
# This survived three reviews in different forms. A claim used to mean "this strategy owns the
# symbol", so the whole account quantity was fair game; then a fix capped by broker attribution but
# fell back to the account quantity when attribution was missing, which is the same bug on the path
# that matters most. The claim's QUANTITY is the cap that holds in every case.

def _owner_fixture(account, claims, attributed=None):
    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    seen = {}

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return dict(claims)

        async def _drop_state(self, sym):
            return None

        async def _save_state(self, sym, st, qty=None):
            return None

        async def _resume(self, session, st, held_qty, panel, slot=None):
            # held_qty is the computed ownership boundary — capture it where it is handed on
            seen["held_qty"] = dict(held_qty)
            from kumo_strategies.runtime.executor.runner import SessionResult
            return SessionResult(session, st.value, False, blocked="captured")

    class Broker:
        def positions(self):
            return dict(account)

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 10.0

    if attributed is not None:
        Broker.strategy_positions = lambda self: dict(attributed)

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def decided_this_session(self, session, slot=None):
            return True

        async def explain(self, session, slot=None):
            return None

        async def tail(self, n=50, kind=None, symbol=None):
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return set(account)

        async def must_liquidate(self, sym):
            return False

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    return r, seen


def test_an_exit_never_reaches_past_what_this_strategy_claimed():
    """MOMENTUM holds 10 AAPL, a manual trade holds 90. The account reports 100. Selling 100 to
    close our 10 liquidates someone else's position."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAPL": 100}, claims={"AAPL": 10},
                             attributed={"AAPL": 10})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"AAPL": 10}, f"would have sold {seen['held_qty']}"


def test_missing_attribution_still_caps_at_the_claim_not_the_account():
    """A broker that cannot attribute per strategy must not widen the exit to the whole account —
    and must not be unable to exit at all either."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAPL": 100}, claims={"AAPL": 10}, attributed=None)
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"AAPL": 10}


def test_a_claim_with_no_quantity_sells_nothing():
    """Legacy rows predate the qty column. Zero must mean no order, not 'unknown, take the lot'."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAPL": 100}, claims={"AAPL": 0}, attributed={"AAPL": 100})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {}


def test_a_broker_position_we_never_recorded_is_adopted_not_orphaned():
    """A buy that filled while the process was dying leaves a real position with no claim. Unclaimed
    reads as foreign, and foreign is never exited by anything — including liquidation."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAPL": 10}, claims={}, attributed={"AAPL": 10})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"AAPL": 10}, "the orphaned position was left unmanaged"


def test_give_back_prices_on_the_current_market_not_yesterdays_close():
    """The session fires at the open, so the last daily bar is yesterday. A position that gapped
    through the give-back level overnight is precisely the one that must exit — entry 10, peak 20,
    yesterday 18, opening at 12 breaches the rule, and reading 18 holds it."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig, MomentumRotationConfig)

    from kumo_strategies.strategies.momentum_rotation.exits import LIVE, TrailState

    class Runner(PgSessionRunner):
        async def _load_state(self, symbols):
            # entry 10, peak 20, observed — so the give-back rule is genuinely armed
            return {"AAA": TrailState(entry_px=10.0, peak_px=20.0, quality=LIVE)}

        async def _save_state(self, sym, st, qty=None):
            return None

    class Broker:
        def positions(self):
            return {"AAA": 10}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 12.0                            # gapped down overnight

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(exits=ExitConfig(give_back_frac=0.5)), broker=Broker())
    exits = aio.run(r._trail_exits({"AAA": 10}, {"AAA": 18.0}))       # stale close says hold
    assert "AAA" in exits, f"gap through the give-back level did not fire: {exits}"


def test_liquidating_flattens_even_when_the_session_already_decided():
    """An operator flipping to LIQUIDATING mid-morning used to be honoured only by the NEXT
    session, and if today had already decided, the resume path replayed the old decision instead.
    The UI said liquidating; the account stayed long."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    sent = []

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return {"AAA": 10, "BBB": 5}

        async def _submit(self, session, enters, exits, held_qty, last, st, *, decision_slot=""):
            sent.append({"enters": enters, "exits": exits})
            return len(exits)

    class Broker:
        def positions(self):
            return {"AAA": 10, "BBB": 5}

        def equity(self):
            return 100_000.0

        def strategy_positions(self):
            return {"AAA": 10, "BBB": 5}

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def decided_this_session(self, session, slot=None):
            return True                      # already decided today

        async def explain(self, session, slot=None):
            return {"detail": {"target_book": ["AAA", "BBB"], "reasons": {}}}

        async def tail(self, n=50, kind=None, symbol=None):
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"AAA", "BBB"}

        async def must_liquidate(self, sym):
            return False

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.LIQUIDATING, "operator"),
               cfg=MomentumRotationConfig(), broker=Broker())
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert sent, "LIQUIDATING did nothing on an already-decided session"
    assert set(sent[0]["exits"]) == {"AAA", "BBB"}
    assert sent[0]["enters"] == ()


def test_a_blacklist_after_the_decision_still_exits_that_symbol():
    """Blacklisting a held name is the operator saying they know something the ranking does not."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    sent = []

    class Runner(PgSessionRunner):
        async def _foreign_claims(self):
            # No other strategy claims anything in these fixtures. Present because
            # production asks for it: a double answering only `_owned` is more
            # forgiving than production and cannot show a cross-strategy sell.
            return {}

        async def _owned(self):
            return {"AAA": 10, "BBB": 5}

        async def _submit(self, session, enters, exits, held_qty, last, st, *, decision_slot=""):
            sent.append(exits)
            return len(exits)

    class Broker:
        def positions(self):
            return {"AAA": 10, "BBB": 5}

        def equity(self):
            return 100_000.0

        def strategy_positions(self):
            return {"AAA": 10, "BBB": 5}

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def decided_this_session(self, session, slot=None):
            return True

        async def explain(self, session, slot=None):
            return {"detail": {"target_book": ["AAA", "BBB"], "reasons": {}}}

        async def tail(self, n=50, kind=None, symbol=None):
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"AAA", "BBB"}

        async def must_liquidate(self, sym):
            return sym == "AAA"

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert sent and sent[0] == ("AAA",), f"blacklisted symbol not exited: {sent}"


def test_an_entry_is_skipped_rather_than_sized_off_a_stale_close():
    """Yesterday's close is wrong by exactly the overnight gap. A skipped entry costs an
    opportunity once; an entry sized on a stale price costs money every time."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(prices={})          # no live price for anything
    aio.run(r._submit("2026-08-04", ("AAA",), (), {}, {"AAA": 10.0}, State.TRADING))
    assert sent == [], "sized an entry off a stale close"


def test_a_position_the_broker_cannot_attribute_is_still_sellable():
    """With the durable cache off, a position reconciled in after a restart comes back without a
    Nautilus strategy id. Reading that as 'we own none of it' made give-back and even LIQUIDATING
    refuse to sell a position we opened — the mirror of the bug attribution was added to fix, and
    just as unexitable. The CLAIM is what caps us; attribution only narrows where it reports."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAA": 10}, claims={"AAA": 10}, attributed={})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"AAA": 10}, f"unexitable after restart: {seen['held_qty']}"


def test_attribution_still_narrows_when_it_reports_the_symbol():
    """The original bug must stay fixed: 10 ours beside a manual 90 sells 10, not 100."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"AAA": 100}, claims={"AAA": 100}, attributed={"AAA": 10})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"AAA": 10}


def test_a_stale_export_is_refused(tmp_path):
    """The staleness rule itself, stated explicitly instead of ridden on by accident.

    Three fixtures below used to hardcode `generated_at`, so they drifted past MAX_META_AGE_HOURS and
    began failing on age while testing checksums and row counts. Giving them a fresh timestamp fixed
    the rot but removed the only coverage the age check had. This test owns it: it states its own age
    rather than depending on what today's date happens to be, so it can neither rot nor be the
    incidental cause of someone else's failure.

    The behaviour matters — a stale book must not drive a session. On 6 Aug the pool was a day behind
    and every exit decision rested on it.
    """
    import hashlib
    import json as _json
    from datetime import timedelta

    from kumo_strategies.runtime.executor.sources.base import SourceError
    from kumo_strategies.runtime.executor.sources.ledger import (
        MAX_META_AGE_HOURS, LedgerBookSource)

    csv = tmp_path / "ledger-book.csv"
    csv.write_text("symbol,open_date,close_date\nAAA,2026-08-01,\n")
    body = csv.read_bytes()
    old = datetime.now(timezone.utc) - timedelta(hours=MAX_META_AGE_HOURS + 1)
    (tmp_path / "ledger-book.meta.json").write_text(_json.dumps({
        "status": "ok", "generated_at": old.strftime("%Y-%m-%dT%H:%M:%SZ"), "row_count": 1,
        "csv_sha256": hashlib.sha256(body).hexdigest()}))

    with pytest.raises(SourceError) as e:
        LedgerBookSource(csv_path=str(csv)).fetch()
    assert "old" in str(e.value).lower() or "stale" in str(e.value).lower()


def test_a_fresh_export_is_accepted(tmp_path):
    """The other half — otherwise the test above passes just as well against a source that refuses
    everything."""
    import hashlib
    import json as _json

    from kumo_strategies.runtime.executor.sources.ledger import LedgerBookSource

    csv = tmp_path / "ledger-book.csv"
    csv.write_text("symbol,open_date,close_date\nAAA,2026-08-01,\n")
    body = csv.read_bytes()
    (tmp_path / "ledger-book.meta.json").write_text(_json.dumps({
        "status": "ok", "generated_at": _fresh_generated_at(), "row_count": 1,
        "csv_sha256": hashlib.sha256(body).hexdigest()}))

    assert LedgerBookSource(csv_path=str(csv)).fetch() is not None


def test_a_csv_that_does_not_match_its_sidecar_is_refused(tmp_path):
    """The CSV and sidecar are separate atomic writes. Reading between them pairs a new book with
    the old manifest, and status/generated_at/schema all validate the wrong file. Same shape as the
    pool sitting four hours behind ledger-tool's fix: each layer internally consistent, pairing wrong."""
    import hashlib
    import json as _json

    import pytest

    from kumo_strategies.runtime.executor.sources.base import SourceError
    from kumo_strategies.runtime.executor.sources.ledger import LedgerBookSource

    csv = tmp_path / "ledger-book.csv"
    csv.write_text("symbol,open_date,close_date\nAAA,2026-08-01,\n")
    meta = {"status": "ok", "generated_at": _fresh_generated_at(), "row_count": 1,
            "csv_sha256": hashlib.sha256(b"a different export entirely").hexdigest()}
    (tmp_path / "ledger-book.meta.json").write_text(_json.dumps(meta))
    with pytest.raises(SourceError) as e:
        LedgerBookSource(csv_path=str(csv)).fetch()
    assert "sidecar" in str(e.value)


def test_a_row_count_mismatch_is_refused(tmp_path):
    import json as _json

    import pytest

    from kumo_strategies.runtime.executor.sources.base import SourceError
    from kumo_strategies.runtime.executor.sources.ledger import LedgerBookSource

    csv = tmp_path / "ledger-book.csv"
    csv.write_text("symbol,open_date,close_date\nAAA,2026-08-01,\n")
    (tmp_path / "ledger-book.meta.json").write_text(
        _json.dumps({"status": "ok", "generated_at": _fresh_generated_at(), "row_count": 975}))
    with pytest.raises(SourceError) as e:
        LedgerBookSource(csv_path=str(csv)).fetch()
    assert "manifest" in str(e.value)


def test_a_recently_confirmed_lot_survives_expiry(tmp_path):
    """The cohort expiry could never keep: long holds he stops posting about. AEM, CGAU, SCCO and
    WPM are held for many months and confirmed by full reviews, but an age rule retires exactly
    those. A confirmation at the corpus edge outranks age."""
    import hashlib
    import json as _json

    from kumo_strategies.runtime.executor.sources.ledger import LedgerBookSource

    csv = tmp_path / "ledger-book.csv"
    csv.write_text(
        "symbol,open_date,close_date,last_confirmed_held\n"
        "OLD,2025-01-01,,2026-07-25\n"        # 580d old, confirmed at the edge -> keep
        "STALE,2025-01-01,,2025-03-01\n"      # 580d old, last confirmed long ago -> expire
        "NEVER,2025-01-01,,\n"                # never confirmable -> expiry decides -> expire
        "RECENT,2026-07-20,,\n"               # young, no confirmation -> expiry keeps it
    )
    body = csv.read_bytes()
    (tmp_path / "ledger-book.meta.json").write_text(_json.dumps({
        "status": "ok", "generated_at": _fresh_generated_at(), "row_count": 4,
        "csv_sha256": hashlib.sha256(body).hexdigest(),
        "latest_full_review": "2026-07-25"}))

    res = LedgerBookSource(csv_path=str(csv)).fetch(
        asof=__import__("datetime").datetime(2026, 8, 4, tzinfo=__import__("datetime").timezone.utc))
    got = set(res.symbols)
    assert "OLD" in got, "a lot confirmed at the corpus edge was retired on age"
    assert "RECENT" in got
    assert "STALE" not in got, "a lot last confirmed months ago should fall back to expiry"
    assert "NEVER" not in got, "an unconfirmable lot past expiry should still expire"


def test_a_failed_source_retries_quickly_rather_than_waiting_out_its_cadence():
    """A failed refresh still stamps last_run, so the cadence clock restarts on failure. With a
    30-minute cadence a transient fault blocks the source for 30 minutes — and the runner treats a
    stale source as a HARD BLOCK on deciding, so it can cost the whole session.

    That is not hypothetical: a missing container mount failed the refresh at 13:05, the cadence
    pushed the retry to 13:35, and the session fires at 13:35."""
    from datetime import datetime, timezone

    from kumo_strategies.runtime.executor.pgjobs import _due

    now = datetime(2026, 8, 4, 13, 20, tzinfo=timezone.utc)
    failed = _spec_row(enabled=True, every_minutes=30, last_status="failed",
                       last_run=datetime(2026, 8, 4, 13, 5, tzinfo=timezone.utc))
    assert _due(failed, now=now) is True, "a failed source waited out its full cadence"

    # ...but not instantly, or a hard-down upstream is hammered every tick
    just_failed = _spec_row(enabled=True, every_minutes=30, last_status="failed",
                            last_run=datetime(2026, 8, 4, 13, 19, 30, tzinfo=timezone.utc))
    assert _due(just_failed, now=now) is False

    # a SUCCESSFUL source keeps its normal cadence
    ok = _spec_row(enabled=True, every_minutes=30, last_status="ok",
                   last_run=datetime(2026, 8, 4, 13, 5, tzinfo=timezone.utc))
    assert _due(ok, now=now) is False


def test_resume_retries_a_failed_submit_but_not_forever():
    """Counting any result row as an attempt — including failures — meant a session that failed for
    an ENVIRONMENTAL reason could never recover. The first live run refused all 8 entries because
    instrument definitions were missing; once that was fixed, resume still said "already decided"
    because those failures looked like attempts."""
    from kumo_strategies.strategies.momentum_rotation.runner import _MAX_SUBMIT_ATTEMPTS

    def attempted(rows):
        results = {}
        for r in rows:
            d = r.get("detail") or {}
            if r["session"] == "S" and r.get("symbol") and d.get("phase") == "result":
                results.setdefault(r["symbol"], []).append(bool(d.get("ok")))
        return {s for s, oks in results.items() if any(oks) or len(oks) >= _MAX_SUBMIT_ATTEMPTS}

    ok = [{"session": "S", "symbol": "A", "detail": {"phase": "result", "ok": True}}]
    assert attempted(ok) == {"A"}, "a successful submit must not be repeated"

    one_fail = [{"session": "S", "symbol": "B", "detail": {"phase": "result", "ok": False}}]
    assert attempted(one_fail) == set(), "a single failure must stay retryable"

    many = [{"session": "S", "symbol": "C", "detail": {"phase": "result", "ok": False}}] * _MAX_SUBMIT_ATTEMPTS
    assert attempted(many) == {"C"}, "a repeatedly failing symbol must stop being retried"


def test_an_adopted_position_uses_the_brokers_real_entry_not_todays_quote():
    """#197 B1. Seeding entry from the current quote is how the trail became fiction. Nautilus
    already carries avg_px_open on every position; the runner simply never asked."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig, MomentumRotationConfig)
    from kumo_strategies.strategies.momentum_rotation.exits import ADOPTED

    saved = {}

    class Runner(PgSessionRunner):
        async def _load_state(self, symbols):
            return {}                                  # nothing on record: must adopt

        async def _save_state(self, sym, st, qty=None):
            saved[sym] = st

    class Broker:
        def positions(self):
            return {"AAA": 10}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 12.0                                # today's quote, far from the real fill

        def position_entries(self):
            return {"AAA": 47.5}                       # what it actually opened at

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(exits=ExitConfig(give_back_frac=0.5)), broker=Broker())
    aio.run(r._trail_exits({"AAA": 10}, {}))
    trail = r._pending_trail["AAA"][0]
    assert trail.entry_px == 47.5, f"seeded from the quote again: {trail}"
    assert trail.quality == ADOPTED, "peak is still unknown — quality must say so"
    assert saved == {}, "the trail must not be persisted before the decision row is won"


def test_adoption_falls_back_to_the_quote_when_the_broker_cannot_attribute():
    """A broker with no per-strategy attribution (a plain REST account view) cannot know the entry.
    Falling back is correct; pretending otherwise is not."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig, MomentumRotationConfig)

    saved = {}

    class Runner(PgSessionRunner):
        async def _load_state(self, symbols):
            return {}

        async def _save_state(self, sym, st, qty=None):
            saved[sym] = st

    class Broker:                                      # no position_entries at all
        def positions(self):
            return {"AAA": 10}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 12.0

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(exits=ExitConfig(give_back_frac=0.5)), broker=Broker())
    aio.run(r._trail_exits({"AAA": 10}, {}))
    assert r._pending_trail["AAA"][0].entry_px == 12.0


def test_the_orphan_sweep_adopts_at_the_brokers_entry_too():
    """Codex caught this on review: the orphan sweep in run() seeded entry from the quote on its own,
    and because it PERSISTS a row, the later _trail_exits found state and never re-adopted. Fixing
    the evaluator path alone left the live path exactly as wrong as before."""
    import asyncio as aio

    saved = {}
    r, _seen = _owner_fixture(account={"AAPL": 10}, claims={}, attributed={"AAPL": 10})

    async def capture(sym, st, qty=None):
        saved[sym] = st
    r._save_state = capture
    type(r.broker).position_entries = lambda self: {"AAPL": 47.5}    # real fill, quote says 10.0

    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert saved["AAPL"].entry_px == 47.5, f"orphan seeded from the quote: {saved.get('AAPL')}"


def test_a_session_blocked_after_the_exits_ran_does_not_advance_the_trail():
    """The loser of a concurrent race, and any session stopped by a data gate, must leave the trail
    alone: it submitted nothing, so counting it against max_hold_days would age positions out of the
    book on days we refused to act."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig, MomentumRotationConfig)
    from kumo_strategies.strategies.momentum_rotation.exits import LIVE, TrailState

    saved = {}

    class Runner(PgSessionRunner):
        async def _load_state(self, symbols):
            return {"AAA": TrailState(entry_px=10.0, peak_px=20.0, quality=LIVE,
                                      sessions_held=3)}

        async def _save_state(self, sym, st, qty=None):
            saved[sym] = st

    class Broker:
        def positions(self):
            return {"AAA": 10}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return 19.0

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(exits=ExitConfig(give_back_frac=0.5)), broker=Broker())
    aio.run(r._trail_exits({"AAA": 10}, {}))

    assert saved == {}, "state was persisted by a session that had not yet won its decision row"
    assert r._pending_trail["AAA"][0].sessions_held == 4, "the advance itself must still be computed"


# -- inverse-vol sizing reaching the LIVE runner (#26) -------------------------------------------
# `max_correlation` and `inverse_vol_sizing` were inert in production: this runner called decide()
# without `corr`/`vol` and then sized every entry at a flat
# `equity * max_deployed_frac / max_positions`, ignoring `dec.weights`. Either omission alone makes
# the flag a no-op that typechecks, deploys and reports no error.
#
# Both halves matter and the second is the one that was missed in the backtest for weeks: passing
# `vol` while still sizing flat leaves the flag just as dead. So these test the SIZE of the order,
# not whether an input was threaded.

def _ivol_cfg():
    from kumo_strategies.strategies.momentum_rotation.config import (
        MomentumRotationConfig, PortfolioConfig)
    return MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=4, inverse_vol_sizing=True))


def test_live_entries_size_from_the_decision_weights():
    """A low-vol name must get more than one equal-weight slot of dollars.

    Book of four, `AAA` weighted 0.40 against an equal 0.25 — so 1.6 slots. At
    equity 100k, max_deployed_frac 0.75 and max_positions 4, a slot is 18,750;
    1.6 slots at $10 is 3,000 shares against the flat 1,875.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(cfg=_ivol_cfg(), prices={"AAA": 10.0},
                              limits={"book_size": 4, "max_deployed_frac": 0.75,
                                      "max_position_notional": 1_000_000.0})
    weights = {"AAA": 0.40, "BBB": 0.20, "CCC": 0.20, "DDD": 0.20}
    aio.run(r._submit("2026-08-04", ("AAA",), (), {"BBB": 1, "CCC": 1, "DDD": 1},
                      {"AAA": 10.0}, State.TRADING, weights=weights))
    assert sent[0].qty == 3000, (
        f"live sizing ignored dec.weights: got {sent[0].qty}, expected 3000. A flat 1875 means the "
        "weights are still being computed and discarded, which is what made inverse_vol_sizing "
        "inert in production")


def test_equal_weights_size_exactly_as_the_flat_budget_did():
    """The no-change guarantee. Every config running today is equal-weighted, and must be untouched.

    Equal weight makes `w = 1/len(book)`, so the multiple is exactly 1.0 and the arithmetic reduces
    to the previous formula rather than merely approximating it.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(cfg=_ivol_cfg(), prices={"AAA": 10.0},
                              limits={"book_size": 4, "max_deployed_frac": 0.75,
                                      "max_position_notional": 1_000_000.0})
    weights = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}
    aio.run(r._submit("2026-08-04", ("AAA",), (), {"BBB": 1, "CCC": 1, "DDD": 1},
                      {"AAA": 10.0}, State.TRADING, weights=weights))
    assert sent[0].qty == 1875, f"equal weight must reduce to the flat slot, got {sent[0].qty}"


def test_max_position_notional_still_binds_after_the_weight_is_applied():
    """A hard risk limit is not negotiable by a sizing rule.

    Without ordering the cap AFTER the weight, a low-vol name earning three slots would breach the
    per-position notional ceiling — the sizing rule would have quietly raised a risk limit.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    r, sent = _submit_fixture(cfg=_ivol_cfg(), prices={"AAA": 10.0},
                              limits={"book_size": 4, "max_deployed_frac": 0.75,
                                      "max_position_notional": 20_000.0})
    weights = {"AAA": 0.70, "BBB": 0.10, "CCC": 0.10, "DDD": 0.10}
    aio.run(r._submit("2026-08-04", ("AAA",), (), {"BBB": 1, "CCC": 1, "DDD": 1},
                      {"AAA": 10.0}, State.TRADING, weights=weights))
    assert sent[0].qty == 2000, (
        f"max_position_notional breached: {sent[0].qty} shares at $10 exceeds the $20k cap")


def test_the_runner_honours_weights_whenever_the_engine_supplies_them():
    """The runner must not decide whether the engine's output matters.

    This test previously asserted the OPPOSITE — that a config without `inverse_vol_sizing` had to
    size flat even when weights were supplied — and called it defence in depth. That was wrong, and
    it encoded the #26 defect as a requirement: a runner second-guessing the config it was handed is
    exactly how `max_correlation` and `inverse_vol_sizing` stayed inert for months.

    The correct separation is that the FLAG controls what `_weights()` returns, and the runner
    consumes whatever it gets. Under equal weight the engine returns `1/len(book)`, so the multiple
    is exactly 1.0 and the flat budget falls out by arithmetic rather than by a conditional. Any
    future weighting scheme is then honoured automatically instead of being silently ignored until
    someone remembers to flip an unrelated flag.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State
    from kumo_strategies.strategies.momentum_rotation.config import (
        MomentumRotationConfig, PortfolioConfig)

    plain = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=4))
    limits = {"book_size": 4, "max_deployed_frac": 0.75,
              "max_position_notional": 1_000_000.0}

    # Equal weights, flag off: must reduce exactly to the flat slot.
    r, sent = _submit_fixture(cfg=plain, prices={"AAA": 10.0}, limits=limits)
    aio.run(r._submit("2026-08-04", ("AAA",), (), {"BBB": 1, "CCC": 1, "DDD": 1},
                      {"AAA": 10.0}, State.TRADING,
                      weights={n: 0.25 for n in ("AAA", "BBB", "CCC", "DDD")}))
    assert sent[0].qty == 1875, f"equal weight must give the flat slot, got {sent[0].qty}"

    # Non-equal weights, same flag-off config: must be honoured, not discarded.
    r2, sent2 = _submit_fixture(cfg=plain, prices={"AAA": 10.0}, limits=limits)
    aio.run(r2._submit("2026-08-04", ("AAA",), (), {"BBB": 1, "CCC": 1, "DDD": 1},
                       {"AAA": 10.0}, State.TRADING,
                       weights={"AAA": 0.40, "BBB": 0.20, "CCC": 0.20, "DDD": 0.20}))
    assert sent2[0].qty == 3000, (
        f"the runner discarded supplied weights because a flag was off: got {sent2[0].qty}. That is "
        "the runner overriding the engine, which is the defect #26 documents.")


# -- the position cap and rotation are in direct conflict -------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason=(
        "REPRODUCED, NOT FIXED (2026-08-21). A book at `max_positions` cannot rotate: it sells N and "
        "every one of the N replacements is refused, because `live = len(held_qty)` still counts the "
        "names just sold. MOMENTUM-002 lost 19 entries this way across seven sessions since "
        "2026-08-06 and has not completed a rotation since. The obvious fix — exclude names being "
        "exited — DIRECTLY CONTRADICTS `test_an_accepted_sell_does_not_free_a_slot_for_a_new_buy`, "
        "which is a deliberate guard: `ok` means accepted, not filled, so a rejected sell would leave "
        "the book over its cap. That trade-off is a risk decision on a live book and is the operator's to "
        "take, not one to make unilaterally from here. strict=True: whoever resolves it must delete "
        "this marker and read why it existed."
    ),
)
def test_a_book_at_its_cap_can_still_ROTATE():
    """MOMENTUM-002 HAS NOT COMPLETED A ROTATION SINCE 2026-08-06.

    Its researched book is `n_hold=8` (cockpit momentum.py:349, "IS the researched configuration, not
    an unvalidated operator override"). Its limits are a bare `RiskLimits()`, so `max_positions` is
    the default — also 8. Two numbers that had to agree, set in two repositories, equal by accident.

    Position-cap refusals per session, from the live journal:

        2026-08-06  6    2026-08-10  2    2026-08-11  1    2026-08-13  1
        2026-08-14  1    2026-08-19  6    2026-08-21  2   <- sold WHD and XLV, entered nothing, 8 -> 6

    Nothing errored. Each refusal journalled a tidy "position cap" and the strategy looked healthy.

    NOT FIXABLE BY RAISING THE CAP: `max_positions` is also the sizing divisor
    (`slot = equity * max_deployed_frac / max_positions`), so raising it silently halves every
    position. One number doing two jobs, and only one of them wants changing.

    THE TWO SHAPES, both real:
      a) exclude names being exited from `live`. Rotation works; a rejected sell leaves the book over
         cap for one session, corrected by the next session's exits.
      b) split the number — a sizing basis (`n_hold`) separate from a cap (`n_hold + headroom`) — so
         the cap can grow without touching position size.
    (b) is the cleaner design and the larger change; (a) is one line and reverses a guard someone
    wrote deliberately.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import State

    held = {f"H{i}": 10 for i in range(8)}                       # exactly at the cap
    r, sent = _submit_fixture(prices={"NEW1": 10.0, "NEW2": 10.0},
                              limits={"max_positions": 8})
    aio.run(r._submit("2026-08-21", ("NEW1", "NEW2"), ("H0", "H1"), held,
                      {"H0": 10.0, "H1": 10.0, "NEW1": 10.0, "NEW2": 10.0}, State.TRADING))
    buys = sorted(s.symbol for s in sent if s.side == "BUY")
    assert buys == ["NEW1", "NEW2"], f"rotation did not complete: bought {buys}"


# -- the claim ledger must not exceed the account --------------------------------------------------
def test_claims_that_exceed_the_account_are_JOURNALLED():
    """THE INVARIANT THE OWNERSHIP RULES EXIST TO MAINTAIN, asserted directly.

    The per-strategy split is unanchored — the broker has ONE net position per symbol and no opinion
    about whose it is (CLAUDE.md; kumo-trading-platform issue 437). So it cannot be repaired by reconciliation and
    must be protected at write time. But every write-time rule we have debated — FIFO by opening lot,
    pro-rata, absence-only retirement — is a mechanism, and we have spent three days on what happens
    when a mechanism is subtly wrong and nothing checks the property it exists to maintain.

    Claims are retired only when a symbol is ENTIRELY absent (`pgrunner`: `owned - set(account)`, a
    set difference with no quantity comparison). So a PARTIAL close leaves the claim at full size:

        BCTROT claims 11, MOMENTUM claims 19, account holds 30.
        An account-level protective stop sells 11 -> account 19, position NOT absent, NO claim retired.
        Claims sum to 30 against an account of 19. BCTROT reads min(19, 11) = 11 and believes it owns
        shares that were sold.

    This asserts the property instead of the rule: whatever mechanism is right, a ledger claiming more
    than the account holds is wrong, and it fires the same session rather than twelve hours later.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import over_claimed

    # the MRVL shape: two strategies, a partial close, no claim retired
    bad = over_claimed(account={"MRVL": 19}, claims_by_strategy={"BCTROT-004": {"MRVL": 11},
                                                                 "MOMENTUM-002": {"MRVL": 19}})
    assert bad == {"MRVL": (30, 19)}, f"did not detect the over-claim: {bad}"


def test_a_ledger_that_agrees_with_the_account_is_silent():
    """A detector that fires on the healthy case is one people learn to ignore."""
    from kumo_strategies.strategies.momentum_rotation.runner import over_claimed

    assert over_claimed(account={"MRVL": 30}, claims_by_strategy={"A": {"MRVL": 11},
                                                                  "B": {"MRVL": 19}}) == {}


def test_claiming_LESS_than_the_account_is_not_an_over_claim():
    """Foreign or manual positions the strategies do not claim are normal and must not alarm."""
    from kumo_strategies.strategies.momentum_rotation.runner import over_claimed

    assert over_claimed(account={"AAPL": 100}, claims_by_strategy={"A": {"AAPL": 10}}) == {}


def test_a_symbol_absent_from_the_account_entirely_is_still_an_over_claim():
    """The one case the absence rule DOES cover — but it must be reported by the same invariant, or
    two mechanisms answer one question and the second is invisible."""
    from kumo_strategies.strategies.momentum_rotation.runner import over_claimed

    assert over_claimed(account={}, claims_by_strategy={"A": {"GONE": 5}}) == {"GONE": (5, 0)}


# -- the dry-run broker must be as UNFORGIVING as production ----------------------------------------
def test_the_dry_run_broker_distinguishes_OWN_positions_from_the_ACCOUNTS():
    """A DRY RUN THAT CANNOT FAIL PROVES NOTHING.

    `DryRunBroker` is the tool a strategy uses to prove itself before it trades, and it offered only
    `positions()` — the account's book. A gateway reading that for OWNERSHIP passes here and then, in
    production, proposes exits against other strategies' holdings. TECHIVOL-005 did exactly that on
    2026-08-21: eight liquidation orders against BCTROT's and MOMENTUM's book, and every test it had
    was green.

    Same flaw as any over-forgiving double, but worse, because this one ships as the dry run.
    """
    from kumo_strategies.runtime.executor.broker import DryRunBroker

    b = DryRunBroker(starting_equity=100_000.0)
    b.adopt_foreign({"AEM": 18, "BDX": 65})          # another strategy's book, on the same account
    assert b.positions() == {"AEM": 18, "BDX": 65}, "the account's book must include foreign holdings"
    assert b.strategy_positions() == {}, (
        "a strategy that has traded nothing must own nothing — this is the read a gateway must use")


def test_the_dry_run_broker_can_report_NO_equity():
    """`equity()` returned a constant, so a strategy that cannot read equity in production reads a
    healthy 100,000 in the dry run. Both of QC345's live failures were equity reading as None."""
    from kumo_strategies.runtime.executor.broker import DryRunBroker

    assert DryRunBroker(starting_equity=None).equity() is None


def test_the_dry_run_broker_prices_symbols_so_SIZING_is_exercised():
    """Without a price there is no sizing, and "sizing yielded 0 shares" is the exact signature of
    QC345's second failure — money, a price, and still nothing to buy."""
    from kumo_strategies.runtime.executor.broker import DryRunBroker

    b = DryRunBroker(prices={"AAPL": 190.0})
    assert b.last_price("AAPL") == 190.0
    assert b.last_price("NOPE") is None, "an unpriced symbol must be None, not a plausible default"


def test_a_dry_run_fill_moves_only_THIS_strategys_book():
    """The foreign holdings must not move when this strategy trades, or the dry run cannot show a
    gateway the difference between its own position and the account's."""
    from kumo_strategies.runtime.executor.broker import DryRunBroker, OrderRequest

    b = DryRunBroker(prices={"AAPL": 190.0})
    b.adopt_foreign({"AEM": 18})
    b.submit(OrderRequest(symbol="AAPL", side="BUY", qty=5, session="2026-08-24",
                          strategy_id="TEST-001"))
    assert b.strategy_positions() == {"AAPL": 5}
    assert b.positions() == {"AEM": 18, "AAPL": 5}, "the account book must include both"


# -- a strategy may NEVER sell another strategy's shares --------------------------------------------
def test_the_attribution_fallback_cannot_reach_ANOTHER_strategys_shares():
    """2026-08-22: "strategies are not supposed to trade among each other."

    THIS ALREADY HAPPENED. On 2026-08-21 MOMENTUM-002 sold 28 WHD that BCTROT-004 had bought the day
    before. The account is genuinely flat (+28 / -28 = 0) so reconcile drift reports nothing and is
    RIGHT to; the split underneath is wrong and no reconciliation can see it.

    It is not a raw account read — pgrunner narrows with `min(account, claim, attribution)`. The hole
    is the middle line:

        q = min(acct_qty, claims.get(s2) or 0)
        if mine:                       # SKIPPED ENTIRELY when attribution is empty
            q = min(q, mine.get(s2))

    Attribution goes missing legitimately: a position reconciled in after a restart comes back with no
    strategy id. The fallback exists so LIQUIDATING cannot strand a position we did open, and that
    reasoning is sound — but it never asks whether ANYONE ELSE claims the symbol.

    Sole claimant, attribution missing  -> the fallback is safe, sell what we claim.
    Two claimants, attribution missing  -> we cannot tell ours from theirs, so we may not take theirs.

    The account quantity minus other strategies' claims is the most we can possibly own. That is
    derivable, needs no attribution, and makes the invariant TRUE rather than merely detected.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import own_ceiling

    # the live WHD case: account 28 (BCTROT's), both lanes claim 28, attribution silent
    assert own_ceiling(acct_qty=28, my_claim=28, other_claims=28) == 0, (
        "sized an exit into another strategy's position — this is the 2026-08-21 WHD incident")


def test_the_sole_claimant_fallback_still_works():
    """The reason the fallback exists. One claimant and no attribution: a position reconciled in after
    a restart is ours, and refusing to sell it means LIQUIDATING cannot wind us down."""
    from kumo_strategies.strategies.momentum_rotation.runner import own_ceiling

    assert own_ceiling(acct_qty=28, my_claim=28, other_claims=0) == 28


def test_we_never_exceed_our_own_claim():
    """A claim is the cap on what we opened. The account holding more is someone else's."""
    from kumo_strategies.strategies.momentum_rotation.runner import own_ceiling

    assert own_ceiling(acct_qty=100, my_claim=10, other_claims=0) == 10


def test_partial_overlap_leaves_us_only_the_remainder():
    """Account 30, they claim 19, we claim 19 — at most 11 can be ours, even though we claim more."""
    from kumo_strategies.strategies.momentum_rotation.runner import own_ceiling

    assert own_ceiling(acct_qty=30, my_claim=19, other_claims=19) == 11


def test_it_never_returns_a_negative():
    """Over-claimed books are real (see `over_claimed`); a negative size would flip a sell into a buy."""
    from kumo_strategies.strategies.momentum_rotation.runner import own_ceiling

    assert own_ceiling(acct_qty=5, my_claim=28, other_claims=28) == 0


def test_a_second_claimant_STOPS_the_exit_end_to_end():
    """The invariant through the real sizing path, not just the helper.

    Reproduces 2026-08-21: the account holds 28 WHD that BCTROT opened, MOMENTUM holds a stale claim
    of 28, and Nautilus attribution is silent. Before this fix MOMENTUM sized an exit of 28 and sold
    BCTROT's shares. It must now size nothing.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation import MomentumRotationConfig

    seen = {}

    class Runner(PgSessionRunner):
        async def _owned(self):
            return {"WHD": 28.0}                     # our stale claim

        async def _foreign_claims(self):
            return {"WHD": 28.0}                     # BCTROT claims it too

        async def _drop_state(self, sym):
            return None

        async def _save_state(self, sym, st, qty=None):
            return None

        async def _resume(self, session, st, held_qty, panel, slot=None):
            from kumo_strategies.runtime.executor.runner import SessionResult
            seen["held_qty"] = dict(held_qty)
            return SessionResult(session, st.value, False, blocked="captured")

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def decided_this_session(self, *a, **k):
            return True                              # short-circuit into _resume

        async def tail(self, *a, **k):
            return []

    class Broker:
        def positions(self):
            return {"WHD": 28}                       # the account: BCTROT's shares

        def strategy_positions(self):
            return {}                                # attribution silent — the restart case

        def equity(self):
            return 100_000.0

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"WHD"}

        async def must_liquidate(self, sym):
            return False

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    aio.run(r.run(pd.DataFrame({"ticker": ["WHD"], "date": [pd.Timestamp("2026-08-21")],
                                "close": [50.0]}), "2026-08-24"))
    assert seen.get("held_qty", {}).get("WHD", 0) == 0, (
        f"sized {seen.get('held_qty')} — would sell another strategy's shares, which is the "
        f"2026-08-21 WHD incident")


def test_foreign_claims_selects_OTHER_strategies_not_our_own():
    """The query, not the override. Every test above stubs `_foreign_claims`, so inverting its filter
    from `!=` to `==` survived them all — it would return OUR claims as foreign, making `own_ceiling`
    subtract our own position from itself and silently refusing every legitimate exit.

    A wrong filter here fails CLOSED rather than open, which is safer but just as broken: the strategy
    stops trading and nothing says why. Same silence as the rest of this week.
    """
    import asyncio as aio
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation import MomentumRotationConfig

    rows = [SimpleNamespace(strategy_id="MOMENTUM-002", symbol="OURS", qty=10.0),
            SimpleNamespace(strategy_id="BCTROT-004", symbol="WHD", qty=28.0),
            SimpleNamespace(strategy_id="QC345-003", symbol="WHD", qty=5.0)]

    class Sess:
        async def execute(self, stmt):
            # Honour the WHERE clause the way the database would, so the FILTER is really under test.
            # Read the operator off the clause rather than string-matching rendered SQL: the rendered
            # form varies by dialect and my first attempt at parsing it silently took the wrong branch,
            # which would have made this test pass for the wrong reason.
            op = getattr(stmt.whereclause.operator, "__name__", "")
            mine = "MOMENTUM-002"
            keep = ([r for r in rows if r.strategy_id != mine] if op == "ne"
                    else [r for r in rows if r.strategy_id == mine])
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: keep))

    @asynccontextmanager
    async def sm():
        yield Sess()

    class Jrn:
        strategy_id = "MOMENTUM-002"

        def sessionmaker(self):
            return sm()

    r = PgSessionRunner(pool=None, journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "a"),
                        cfg=MomentumRotationConfig(), broker=None, strategy_id="MOMENTUM-002")
    got = aio.run(r._foreign_claims())
    assert got == {"WHD": 33.0}, f"expected other strategies' WHD claims summed, got {got}"
    assert "OURS" not in got, "returned our OWN claim as foreign — the filter is inverted"


# -- every journal write must carry the session's SLOT ----------------------------------------------
def test_no_journal_write_in_pgrunner_omits_the_slot():
    """EVERY ORDER/RISK/ERROR ROW WAS FILED UNDER THE WRONG SLOT. Verified in the live journal,
    MOMENTUM-002 on 2026-08-19:

        09:35:00  open+5m     decision   TRADING: hold 8 · enter 3 · exit 2
        11:40:00  open+130m   decision   TRADING: hold 8 · enter 2 · exit 2
        11:40:00  open+5m     order      SELL 933 FSM: submitting        <- actually open+130m
        12:27:00  open+177m   decision   TRADING: hold 8 · enter 2 · exit 2
        12:27:00  open+5m     order      SELL 933 FSM: submitting        <- actually open+177m

    The DECISION write receives a real slot; every other write passes `session=` and no `slot=`, and
    `PgJournal.write` defaults to DEFAULT_SLOT. So the two halves of one session disagree inside one
    table, and three different slots share one label.

    THE COST IS A WRONG CONCLUSION, not a wrong label. Reading `select slot, kind, count(*)` shows
    `open+130m: decision 1, order 0`, and a cockpit session concluded the extra slots decided and
    formed nothing. They did not: open+177m submitted FSM 933 and VCTR 88, both FILLED, +$866.00 and
    +$795.52 realised at the broker. The slot-outcome detectors are being built on this column.

    ASSERTED OVER THE AST, and as a rule rather than per call site: a `slot=` that is optional and
    defaulted will be forgotten again — the default is exactly what made this invisible for weeks.
    """
    import ast
    import pathlib

    src = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "kumo_strategies" / "strategies" / \
        "momentum_rotation" / "runner.py"
    offenders = []
    for node in ast.walk(ast.parse(src.read_text())):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "write"
                and isinstance(f.value, ast.Attribute) and f.value.attr == "journal"):
            continue
        if not any(k.arg == "slot" for k in node.keywords):
            offenders.append(node.lineno)
    assert not offenders, (
        f"journal writes with no `slot=` at lines {offenders} — they land on DEFAULT_SLOT and file "
        f"the row under a slot the session never ran, which is how three slots came to share one "
        f"label on 2026-08-19")


def test_a_journal_row_carries_the_SESSIONS_slot_not_the_default():
    """The AST guard proves no call site omits `slot=`. This proves the value that arrives is the
    SESSION'S slot rather than the default — which is the actual 2026-08-19 defect: every row said
    `open+5m` while the session was running open+130m."""
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation import MomentumRotationConfig

    rows = []

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, kind, summary, *, session, detail=None, symbol=None, slot=None):
            rows.append(slot)
            return len(rows)

        async def decided_this_session(self, *a, **k):
            return False

        async def tail(self, *a, **k):
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return set()

        async def must_liquidate(self, sym):
            return False

    class Broker:
        def positions(self):
            return {}

        def strategy_positions(self):
            return {}

        def equity(self):
            return 100_000.0

    class Runner(PgSessionRunner):
        async def _owned(self):
            return {}

        async def _foreign_claims(self):
            return {}

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    # A panel the runner can actually consume — it reads volume for liquidity. A thinner frame
    # raised KeyError before reaching a single journal write, so the assertion below was never
    # exercised and the test passed nothing while looking like a test.
    panel = pd.DataFrame({"ticker": ["A"] * 3,
                          "date": pd.bdate_range("2026-08-17", periods=3),
                          "open": [10.0, 10.1, 10.2], "high": [10.2, 10.3, 10.4],
                          "low": [9.8, 9.9, 10.0], "close": [10.0, 10.1, 10.2],
                          "volume": [1_000_000.0] * 3})
    aio.run(r.run(panel, "2026-08-19", slot="open+130m"))
    assert rows, "the session journalled nothing, so this asserts nothing"
    wrong = [s for s in rows if s != "open+130m"]
    assert not wrong, (
        f"{len(wrong)} of {len(rows)} rows filed under {set(wrong)} instead of the session's "
        f"open+130m — this is the 2026-08-19 defect")


def test_an_EMPTY_account_read_does_not_retire_every_claim():
    """FAILURE MUST NOT BE INFERRED AS ABSENCE (kumo-trading-platform, 2026-08-22).

    An Alpaca 503 at 07:51 ET made Nautilus mark ELEVEN protective stops REJECTED while all eleven
    rested untouched at the broker, from an asymmetry inside ONE reconciliation pass:

        Skipping position reconciliation for AEM.XNYS: failed to query venue   failure -> UNKNOWN
        Reconciling PROT-SELL-BDX-...: not found at venue, marking as REJECTED failure -> ABSENT

    Same outage, same pass, opposite conclusions from the identical error.

    THIS REPO HAS THE SAME SHAPE, and it is destructive rather than cosmetic. `account` comes from the
    Nautilus CACHE, which after a restart is filled by reconciliation — and reconciliation demonstrably
    SKIPS symbols when the venue read fails. A skipped symbol is absent from the cache, absent from
    `account`, and `stale_claims = owned - set(account)` RETIRES ITS CLAIM. The strategy then believes
    it owns nothing it actually holds, and the retirement is written to Postgres, so it outlives the
    outage.

    A restart during a venue wobble is exactly Monday's shape.

    THE RULE: retire on POSITIVE EVIDENCE of absence, never on the absence of evidence. An account read
    that came back completely empty while claims exist is indistinguishable from a failed one, so it
    retires nothing and says so.
    """
    import asyncio as aio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation import MomentumRotationConfig

    dropped, rows = [], []

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, kind, summary, **kw):
            rows.append((kind, summary))
            return len(rows)

        async def decided_this_session(self, *a, **k):
            return True

        async def tail(self, *a, **k):
            return []

    class Pool:
        async def sources(self):
            return []

        async def symbols(self):
            return {"AEM"}

        async def must_liquidate(self, sym):
            return False

    class Broker:
        def positions(self):
            return {}                      # the cache after a failed reconciliation

        def strategy_positions(self):
            return {}

        def equity(self):
            return 100_000.0

    class Runner(PgSessionRunner):
        async def _owned(self):
            return {"AEM": 9.0, "BDX": 55.0}       # we hold these; the broker read says nothing

        async def _foreign_claims(self):
            return {}

        async def _drop_state(self, sym):
            dropped.append(sym)

        async def _save_state(self, sym, st, qty=None):
            return None

        async def _resume(self, session, st, held_qty, panel, slot=None):
            from kumo_strategies.runtime.executor.runner import SessionResult
            return SessionResult(session, st.value, False, blocked="captured")

    r = Runner(strategy_id="MOMENTUM-002", pool=Pool(), journal=Jrn(), lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(), broker=Broker())
    panel = pd.DataFrame({"ticker": ["AEM"] * 3, "date": pd.bdate_range("2026-08-19", periods=3),
                          "open": [1.0] * 3, "high": [1.0] * 3, "low": [1.0] * 3,
                          "close": [1.0] * 3, "volume": [1e6] * 3})
    aio.run(r.run(panel, "2026-08-24"))

    assert dropped == [], (
        f"retired {dropped} on an account read that returned NOTHING — indistinguishable from a failed "
        f"read, and the retirement is written to Postgres so it outlives the outage")
    # THE SPECIFIC MESSAGE, not merely "some row mentions claims". The loose version passed when the
    # guard was mutated to write nothing at all, because an unrelated risk row happened to contain the
    # word — an assertion satisfied by adjacent noise is not an assertion.
    assert any("indistinguishable from a failed one" in s for _, s in rows), (
        f"retired nothing and said nothing — silence is how this stays invisible. Rows: "
        f"{[s for _, s in rows]}")


# -- a read-only wrapper must degrade WRITES, never reads --------------------------------------------
def test_the_readonly_wrapper_exposes_every_READ_the_wrapped_broker_does():
    """FOURTH INSTANCE OF ONE PATTERN TODAY, AND THE ONLY ONE THAT RUNS IN PRODUCTION.

    `ReadOnlyBroker` is an explicit whitelist with no `__getattr__`, so it silently drops five methods
    of the broker it wraps — `feed`, `instrument_ids`, `last_price`, `position_entries`,
    `strategy_positions`. It conformed to the old four-method `Broker` protocol while being strictly
    less capable than the object it stands in for: it satisfied the contract and broke the code.

    WHAT ACTUALLY HAPPENED, from pgrunner's own getattr sites:

        strategy_positions -> None -> `mine = None` -> ADOPTION NEVER RUNS. An unclaimed position "is
            read as foreign, gets leaving-them-alone, and is never exited by anything, INCLUDING
            liquidation" — that comment is three lines from the call.
        last_price -> None -> every trail evaluated against yesterday's close. "A position that gapped
            through the give-back level overnight is exactly the one that needs exiting. Entry 10,
            peak 20, yesterday 18, opening at 12: the rule is breached, and reading 18 holds it
            anyway."
        position_entries -> None -> entry attribution lost, falls back to the quote.

    Every one of those biases in the direction that HIDES exits.

    AND IT CONTRADICTS ITS OWN DOCSTRING, which is how this should have been found faster: "SHADOW is
    meant to compute exactly what TRADING would, minus the submit." Two derivations of one fact — the
    docstring's claim of equivalence and the method list — and the docstring was the one telling the
    truth about the intent.

    THE RULE: a read-only wrapper degrades WRITES. Anything else it drops is a capability the caller
    will silently do without, and `getattr(broker, x, None)` guards turn that into wrong answers rather
    than errors.
    """
    from kumo_strategies.runtime.executor.broker import ReadOnlyBroker
    from kumo_strategies.runtime.nautilus.broker import NautilusBroker

    writes = {"submit", "exit"}
    inner_reads = {n for n in dir(NautilusBroker) if not n.startswith("_")} - writes
    # Wrap a REAL broker. An earlier version wrapped `object()`, which has none of these to forward —
    # the test would then fail whatever the wrapper did, which is a test that cannot distinguish the
    # fix from the defect.
    wrapper = ReadOnlyBroker(NautilusBroker.__new__(NautilusBroker))
    missing = sorted(r for r in inner_reads if not hasattr(wrapper, r))
    assert not missing, (
        f"the read-only wrapper drops {missing} — each becomes None at a `getattr(broker, x, None)` "
        f"site and the caller computes a different, quieter answer instead of failing")


def test_the_readonly_wrapper_still_refuses_to_SUBMIT():
    """Forwarding reads must not forward writes — that is the one thing it exists to stop."""
    from kumo_strategies.runtime.executor.broker import OrderRequest, ReadOnlyBroker

    class Inner:
        def submit(self, req):
            raise AssertionError("the wrapper forwarded a SUBMIT to the real broker")

    res = ReadOnlyBroker(Inner()).submit(
        OrderRequest(symbol="AAA", side="BUY", qty=1, session="2026-08-24", strategy_id="T-000"))
    assert getattr(res, "ok", True) is False, "a read-only wrapper reported a submit as successful"


def test_a_forwarded_read_reaches_the_WRAPPED_broker():
    """Exposing the attribute is not enough — it has to return the inner broker's answer."""
    from kumo_strategies.runtime.executor.broker import ReadOnlyBroker

    class Inner:
        def strategy_positions(self):
            return {"AEM": 9}

        def last_price(self, sym, **kw):
            return 198.95

    ro = ReadOnlyBroker(Inner())
    assert ro.strategy_positions() == {"AEM": 9}
    assert ro.last_price("AEM") == 198.95


def test_getattr_refuses_a_WRITE_name_even_when_called_directly():
    """`_WRITES` is unreachable through ordinary use — normal lookup finds the explicit `submit` and
    `exit`, so `__getattr__` is never consulted for them. That makes it exactly the kind of inert
    defensive code this codebase keeps paying for, so it is tested directly rather than trusted.

    It stops being unreachable the moment someone deletes or renames one of those methods, at which
    point writes would forward silently to the real broker.
    """
    import pytest

    from kumo_strategies.runtime.executor.broker import ReadOnlyBroker

    class Inner:
        def submit(self, req):
            raise AssertionError("forwarded a submit")

    ro = ReadOnlyBroker(Inner())
    with pytest.raises(AttributeError):
        ReadOnlyBroker.__getattr__(ro, "submit")
    with pytest.raises(AttributeError):
        ReadOnlyBroker.__getattr__(ro, "exit")


# -- a short position is never sold into ----------------------------------------------------------

def test_a_SHORT_account_position_is_never_sold_into():
    """Asked by kumo-trading-platform on 2026-08-23: can any path emit a SELL larger than the position held?

    Their finding was a real short on the paper account — `sell_short 23 PENG @ 47.56` on 2026-07-28,
    covered six days later for -$141.22 — with cockpit's `SHORT_PERMITTED` an empty frozenset. This
    repo did not place it: on 2026-07-28 the repository contained exactly one commit, the scaffold.
    The first strategy adapter landed 2026-08-02 and both brokers on 2026-08-04.

    But the property is worth pinning rather than left emergent, because until f7fa96e (2026-08-22)
    the arithmetic went the other way. `_sum_positions` summed UNSIGNED, so a -23 short READ AS +23
    long, and the runner would have sized an exit of 23 against it — taking the position to -46.
    A short from ANY source would have been deepened by this repo, silently, on every rebalance.

    Now the sign survives into `own_ceiling`, which floors at zero, so `held_qty` never receives the
    symbol and `_submit` skips it on `q <= 0`. A short is unexitable here rather than deepened, which
    is the correct end state: this repo is long-only, so a short is someone else's position by
    definition and reaching into it is the whole failure class `own_ceiling` exists for.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import own_ceiling

    assert own_ceiling(acct_qty=-23, my_claim=23, other_claims=0) == 0, (
        "a SHORT account position produced a positive sell quantity — this repo would deepen a short "
        "it did not open")
    for acct in (-1, -23, -1000):
        assert own_ceiling(acct, my_claim=1000, other_claims=0) == 0
        assert own_ceiling(acct, my_claim=0, other_claims=0) == 0


def test_an_exit_can_never_exceed_what_the_ACCOUNT_actually_holds():
    """The other half of the same question: a SELL sized from a stale or assumed quantity.

    `held_qty` is built by intersecting the BROKER'S account with this strategy's claims and clamping
    with `own_ceiling`, so the claim can only ever narrow the account figure, never exceed it. No
    lane sizes an exit from its own bookkeeping: `_held` is a `set[str]` in every adapter — symbols,
    never quantities — and `qc27_runner` reads `strategy_positions()`, which is Nautilus's Cache.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import own_ceiling

    # A claim far larger than the account — a stale claim after a manual partial sale, say.
    assert own_ceiling(acct_qty=10, my_claim=500, other_claims=0) == 10
    # Another strategy holds most of it.
    assert own_ceiling(acct_qty=100, my_claim=100, other_claims=90) == 10
    # Foreign claims exceeding the account cannot make this negative and cannot wrap.
    assert own_ceiling(acct_qty=100, my_claim=100, other_claims=500) == 0


# -- claims retire on QUANTITY, not on presence ----------------------------------------------------

def test_a_claim_survives_a_flat_that_happened_between_reconciles():
    """The BETA freeze, found live by kumo-trading-platform at 23:14 ET on 2026-08-22.

        2026-08-20 13:35   buy  58 + 19 @ 26.10    ->  77 held   MOMENTUM-002 claims 77
        2026-08-20 15:35   sell 2+13+31+18+6+1+5+1 ->   0 held   FLAT
        2026-08-20 16:00   buy  36 + 43 @ 25.24    ->  79 held   BCTROT-004 claims 79

    The position went flat at 15:35 and reopened 25 minutes later, so BETA was never ABSENT from a
    session-start account read. Claims retire on `owned - set(account)` — a set difference with no
    quantity comparison — so MOMENTUM's 77 survived a position it no longer had any part of.

    THE CONSEQUENCE IS NOT AN OVER-SELL. `own_ceiling` already prevents that, correctly. It is a
    FREEZE, and it is the opposite risk:

        BCTROT-004   own_ceiling(79, my_claim=79, other=77) = 2
        MOMENTUM-002 own_ceiling(79, my_claim=77, other=79) = 0
                                                    TOTAL     2 sellable of 79 held

    97% of a real position cannot be exited by anyone, on any path including LIQUIDATING, because
    each lane's ceiling subtracts the other's stale claim. BETA is also one of the positions going
    into Monday's open unprotected: no stop, and no exit either.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import own_ceiling, retire_claims

    account = {"BETA": 79}
    assert own_ceiling(79, 79, 77) == 2 and own_ceiling(79, 77, 79) == 0, "the live freeze"

    # MOMENTUM's own attribution positively reports it holds no BETA — it holds other things, so
    # attribution is working and this is evidence rather than the absence of it.
    # ...and BCTROT-004 positively claims the whole 79, which is what makes our silence about BETA
    # mean something rather than nothing. See the mixed-restart test below.
    retired = retire_claims(owned={"BETA", "AEM"}, account=account | {"AEM": 5},
                            mine={"AEM": 5}, foreign={"BETA": 79})
    assert "BETA" in retired, (
        "a claim on a position this strategy provably does not hold any of must retire, or the "
        "other lane stays frozen out of its own shares")
    assert "AEM" not in retired, "a claim attribution CONFIRMS must never retire"


def test_attribution_that_is_SILENT_retires_nothing():
    """The rule is unchanged: positive evidence of absence, never the absence of evidence.

    `mine` is legitimately empty for a position reconciled in from the broker after a restart — it
    comes back without a strategy id. Treating empty attribution as "this strategy holds none of
    anything" would retire every claim on the account during exactly the window where the strategy
    can least afford to disown its positions.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    for silent in ({}, None):
        assert retire_claims(owned={"BETA"}, account={"BETA": 79}, mine=silent) == [], (
            f"attribution of {silent!r} is not evidence that the claim is dead")


def test_an_EMPTY_account_read_still_retires_nothing():
    """Unchanged guard, restated here because the new rule must not create a second way past it: an
    account read that came back completely empty is indistinguishable from a failed one."""
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    assert retire_claims(owned={"BETA", "AEM"}, account={}, mine={"AEM": 5}) == []


def test_a_symbol_ABSENT_from_the_account_still_retires():
    """The original rule survives — WHD and XLV are cockpit's benign case: 0 held means the symbol is
    genuinely absent, so the set difference retires them at the next session-start reconcile."""
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    assert retire_claims(owned={"WHD", "XLV", "AEM"}, account={"AEM": 5}, mine={"AEM": 5},
                         foreign={}) == ["WHD", "XLV"]


def test_a_MIXED_attribution_restart_disowns_NOTHING():
    """The defect kumo-trading-platform found in e87465a before it shipped. This is the fixture that was
    missing, and its absence is why the mutation bites passed.

    A restart reconciles positions in from the broker WITHOUT a strategy id
    (`nautilus/broker.py`: "a position reconciled in from outside will not carry our id"), so they
    are absent from `mine`. If the lane then opens ONE new position, `mine` is NON-EMPTY and
    INCOMPLETE — and `if mine:` guards only the EMPTY case:

        owned   = six real positions
        mine    = {NEW: 5}
        -> retired ALL SIX, written to Postgres, durable

    Then `own_ceiling(acct, my_claim=0, other)` is 0 for every one and the lane cannot exit ANY of
    its book on any path including LIQUIDATING. Six frozen positions instead of the one being fixed —
    the freeze arriving through the fix for the freeze. And a restart is exactly what shipping this
    requires.

    ATTRIBUTION NOT NAMING A SYMBOL IS EVIDENCE OF NOTHING. The id is missing for a reason unrelated
    to ownership. That is already the rule three lines away, at the sizing site, whose comment
    predates this defect and describes it: "Absence is not evidence of zero: with the durable cache
    off, a position reconciled in from the broker after a restart comes back without a strategy id."

    Two derivations of one fact — does attribution's silence about a symbol mean we hold none of it —
    answered opposite ways in one file. The sizing answer was the careful one.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    owned = {"AEM", "AMGN", "BDX", "BETA", "CGAU", "WPM"}
    account = {"AEM": 18, "AMGN": 8, "BDX": 65, "BETA": 79, "CGAU": 174, "WPM": 26, "NEW": 5}
    # No other lane claims any of these, so nothing is positively somebody else's.
    assert retire_claims(owned, account, mine={"NEW": 5}, foreign={}) == [], (
        "a partial attribution disowned positions the lane genuinely holds; a restart produces "
        "exactly this state")


def test_silence_becomes_evidence_only_when_ANOTHER_lane_positively_claims_the_position():
    """What actually distinguishes tonight's BETA from a restart, and it is neither book-level
    completeness nor a bigger `mine`.

    BETA: the account holds 79, attribution does not name it for us, AND BCTROT-004 claims the whole
    79. Someone else positively owns it, so our claim is provably dead.

    Restart: the account holds 18 AEM, attribution does not name it, and NOBODY else claims it. That
    is a position with no owner on record — which is what a reconciled-in position looks like, and it
    is very likely ours.

    Both cases have non-empty, incomplete `mine`. The discriminator is not how much attribution says;
    it is whether a DIFFERENT lane accounts for the position. Cockpit's suggested
    `set(mine) >= set(account) & owned` cannot work here: BETA is precisely the symbol missing from
    `mine`, so requiring coverage of it makes the test false in exactly the case it must fire.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    account = {"BETA": 79, "AEM": 18}
    mine = {"AEM": 18}

    assert retire_claims({"BETA", "AEM"}, account, mine, foreign={"BETA": 79}) == ["BETA"]
    assert retire_claims({"BETA", "AEM"}, account, mine, foreign={"BETA": 40}) == [], (
        "a partial foreign claim does not account for the position, so our claim is not provably "
        "dead — the safe direction is to keep it and let `over_claimed` alarm")


def test_retirement_returns_ONLY_this_strategys_claims():
    """`foreign` is read to interpret evidence, never to act on another lane's ledger.

    The safety property is unchanged and now needs stating as behaviour rather than as a signature,
    since another lane's claims are in scope: whatever the evidence, the result is a subset of what
    THIS strategy claims. Retiring our own can only reduce what we may sell; retiring someone else's
    would raise our ceiling into their position.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    owned = {"BETA"}
    out = retire_claims(owned, {"BETA": 79, "OTHER": 50}, {"X": 1},
                        foreign={"BETA": 79, "OTHER": 50})
    assert set(out) <= owned, f"retired {out}, which is not a subset of this strategy's claims"


def test_a_position_PRESENT_but_netting_to_zero_retires():
    """The WHD residue, from kumo-trading-platform's read of the live cache legs on 2026-08-23:

        WHD.XNYS  BCTROT-004    LONG   28    signed +28    both OPEN
        WHD.XNYS  MOMENTUM-002  SHORT  28    signed -28
        XLV.ARCX  both lanes    FLAT    0                  both CLOSED

    `positions_open()` excludes flat positions, so XLV is ABSENT from the account map and the set
    difference retires it. WHD has two OPEN legs, so it is PRESENT — as a key whose signed net is 0.

    PRESENT-WITH-ZERO IS NEITHER ABSENT NOR HELD, and retirement tested key MEMBERSHIP. The claim
    stayed at 28/28 against a broker holding none, so OVER-CLAIMED WHD would fire forever and never
    resolve. It is the residue a cross-strategy sell leaves behind — the WHD incident's own footprint.

    Not a trading risk: `own_ceiling(0, 28, 28)` is 0 for both lanes and `held_qty` requires `q > 0`,
    so no order can form. The damage is a permanent false alarm and a ledger that cannot converge —
    and an alarm that never clears is one nobody reads on the morning it means something.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    account = {"WHD": 0, "AEM": 18}
    assert "WHD" in account, "the fixture must reproduce PRESENT-with-zero, not absent"
    assert retire_claims({"WHD", "AEM"}, account, mine={"WHD": -28, "AEM": 18},
                         foreign={"WHD": 28}) == ["WHD"]
    # ...and on the ACCOUNT evidence alone. With a foreign claim present the disown clause also
    # retires it, so the first assertion passes even under key-membership `absent` — it did, under a
    # mutation bite, which is how this second one came to exist.
    assert retire_claims({"WHD", "AEM"}, account, mine={"WHD": -28, "AEM": 18},
                         foreign={}) == ["WHD"], (
        "present-with-zero must retire because the account holds none of it, not because another "
        "lane happens to claim it")
    # ...and with attribution OFF ENTIRELY, which is the only shape that reaches `absent` alone.
    # The two assertions above both survived a key-membership mutation: with `mine` truthy the
    # symbol falls through to `foreign >= account`, and `0 >= 0` is true, so the disown clause
    # retired it for an unrelated reason. Two clauses each hiding the other's mutation — the guard
    # has to bypass one of them to bind the other.
    assert retire_claims({"WHD", "AEM"}, account, mine=None, foreign={}) == ["WHD"], (
        "with no attribution at all, only the account predicate can retire this — and a rule "
        "testing key membership leaves the claim alive forever")


def test_a_NEGATIVE_net_is_not_something_a_long_only_lane_can_own():
    """The same predicate, one step further. These lanes are long-only, so a claim on a net short is
    dead by definition — keeping it produces the identical never-clearing alarm as the zero case."""
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    assert retire_claims({"WHD"}, {"WHD": -28}, mine={"WHD": -28}, foreign={}) == ["WHD"]


def test_attribution_reporting_a_SHORT_is_not_evidence_of_OWNERSHIP():
    """kumo-trading-platform's question, and I agree with their instinct: `> 0`, not truthy.

    A long-only lane holding a short leg holds no shares to preserve. With a truthiness test, an
    attributed -28 reads as "we hold it" and protects a claim that cannot be true — the same
    inversion as the account-side zero, on the other input.

    Here the account nets POSITIVE (another party is long more than we are short), so the zero rule
    above does not reach it; only the attribution predicate does.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    assert retire_claims({"X"}, {"X": 50}, mine={"X": -28}, foreign={"X": 50}) == ["X"]


def test_the_foreign_clause_needs_a_POSITIVE_foreign_claim():
    """Where the foreign clause's boundary actually is, since `absent` now guards the other side.

    `foreign >= account` is trivially satisfied when the account quantity is 0 — including by a
    foreign claim of 0. That is unreachable now, because `absent` removes every non-positive account
    before this loop, so the reachable boundary is the one asserted here: a positive account, silence
    from our attribution, and a foreign claim that does or does not COVER the position.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    # Positive account, silence, and NO foreign claim: the reconciled-in restart case. Keep.
    assert retire_claims({"AEM"}, {"AEM": 18}, mine={"NEW": 5}, foreign={}) == []
    assert retire_claims({"AEM"}, {"AEM": 18}, mine={"NEW": 5}, foreign={"AEM": 0}) == []
    # A foreign claim that does not cover the whole position is not evidence either.
    assert retire_claims({"AEM"}, {"AEM": 18}, mine={"NEW": 5}, foreign={"AEM": 17}) == []
    assert retire_claims({"AEM"}, {"AEM": 18}, mine={"NEW": 5}, foreign={"AEM": 18}) == ["AEM"]


# ==================================================================================================
# A CLAIM THE CACHE CONTRADICTS (kumo-trading-platform issue 692, measured on an Alpaca paper instance 2026-08-28 15:40 ET)
# ==================================================================================================

def test_an_exit_is_refused_when_the_venue_attributes_us_NONE_of_it():
    """MEASURED. BDX on paper:

        claims ledger   MOMENTUM 45 / BCTROT 10
        Nautilus cache  MOMENTUM 55 / BCTROT FLAT

    Totals agree, the SPLIT disagrees. `own_ceiling` cannot see it — `min(10, 55-45)` is 10,
    self-consistent inside a ledger that is itself wrong — so BCTROT sized an exit for 10 shares it
    does not hold, and the sell was a cross-strategy reach VIA the claims ledger. MOMENTUM's
    trailing stop blocked it, and that block is the only reason anyone saw it.

    A strategy may never sell another strategy's shares. When the two records disagree about whose
    they are, the answer is not to pick the one that lets us trade.
    """
    import asyncio as aio

    r, seen = _owner_fixture(account={"BDX": 55}, claims={"BDX": 10},
                             attributed={"OTHER": 3})     # populated, and BDX is not in it
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {}, (
        f"sized an exit off a claim the venue contradicts: {seen['held_qty']}")


def test_the_refusal_is_REPORTED_not_just_silent():
    """A refusal nobody can see is a position that quietly stops being exitable. The divergence is
    also the more important fact: the ledger and the cache disagree about who owns what."""
    import asyncio as aio

    r, _ = _owner_fixture(account={"BDX": 55}, claims={"BDX": 10}, attributed={"OTHER": 3})
    rows = []
    real = r.journal.write

    async def _spy(kind, summary, **kw):
        rows.append(str(summary))
        return await real(kind, summary, **kw)

    r.journal.write = _spy
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    said = " ".join(rows)
    assert "BDX" in said and "contradicts" in said, (
        f"the divergence was absorbed silently. Journal said: {said}")


def test_an_EMPTY_attribution_still_trusts_the_claim():
    """THE GUARANTEE THAT MUST SURVIVE, and the reason this is keyed on `mine` being POPULATED.

    A position reconciled in after a restart carries no strategy id, so `mine` is empty while the
    position is real and ours. Treating that as "we own none of it" would make give-back and even
    LIQUIDATING refuse to sell something we opened — the mirror bug, and just as unexitable.
    """
    import asyncio as aio

    r, seen = _owner_fixture(account={"BDX": 55}, claims={"BDX": 10}, attributed={})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"BDX": 10}, "an unavailable attribution was read as evidence of zero"


def test_an_AGREEING_attribution_is_unaffected():
    """The control. A refusal that fired whenever attribution was present would pass the assertions
    above and stop every exit in the system."""
    import asyncio as aio

    r, seen = _owner_fixture(account={"BDX": 55}, claims={"BDX": 10}, attributed={"BDX": 10})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert seen["held_qty"] == {"BDX": 10}


# ==================================================================================================
# RETIRING A CLAIM DESTROYS DURABLE STATE — record it first (kumo-trading-platform issue 635)
# ==================================================================================================

def _retire_fixture(account, claims, attributed=None, trails=None, load_raises=False):
    """An owner fixture whose trail read can be controlled, plus a journal spy."""
    from kumo_strategies.strategies.momentum_rotation.exits import TrailState

    r, seen = _owner_fixture(account=account, claims=claims, attributed=attributed)
    rows = []
    real = r.journal.write

    async def _spy(kind, summary, **kw):
        rows.append((str(summary), kw.get("detail") or {}))
        return await real(kind, summary, **kw)

    r.journal.write = _spy

    async def _load(symbols):
        if load_raises:
            raise RuntimeError("postgres unreachable")
        return {s: TrailState(entry_px=100.0, peak_px=140.0, opened_session=None,
                              sessions_held=7, sessions_since_high=3, quality="observed")
                for s in symbols if s in (trails or {})}

    r._load_state = _load
    dropped = []
    r._drop_state = lambda sym: dropped.append(sym) or _noop()
    return r, rows, dropped


async def _noop():
    return None


def test_a_retired_claim_records_the_trail_it_DESTROYS():
    """MEASURED RISK, kumo-trading-platform issue 635. `NautilusBroker.positions()` nets, and reconciliation invents
    mirror SHORT positions in the cache — so a real long cancels to ZERO, zero is "account holds
    none of it", and the claim for a position we really hold is retired.

    `_drop_state` is a hard DELETE of the entry price, the running peak, the sessions-held counters
    and the quality flag. This row used to carry only the symbol NAMES, so a wrong retirement was
    unrecoverable: nothing anywhere held what had been deleted.

    Live at time of writing: TECHIVOL-005 holds RBRK 14 against a net-zero symbol — one trail, due
    to be deleted at that lane's next session.
    """
    import asyncio as aio

    r, rows, dropped = _retire_fixture(account={"RBRK": 0}, claims={"RBRK": 14},
                                       attributed={}, trails={"RBRK": True})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert "RBRK" in dropped, "the fixture did not actually retire anything"

    released = [d for s, d in rows if "released" in s]
    assert released, "a claim was destroyed and no row said so"
    state = released[0].get("retired_state", {}).get("RBRK")
    assert state, "the trail was deleted and not recorded — the retirement is unrecoverable"
    assert state["entry"] == 100.0 and state["peak"] == 140.0
    assert state["sessions_held"] == 7 and state["quality"] == "observed"


def test_the_row_records_the_EVIDENCE_the_retirement_rested_on():
    """The account quantity is what the decision was made from, and the first thing to check when a
    retirement turns out wrong. Without it the row says what was destroyed but not why."""
    import asyncio as aio

    r, rows, _ = _retire_fixture(account={"RBRK": 0}, claims={"RBRK": 14},
                                 attributed={}, trails={"RBRK": True})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    released = [d for s, d in rows if "released" in s]
    assert released[0].get("account_qty", {}).get("RBRK") == 0.0


def test_a_FAILED_trail_read_does_not_break_the_session():
    """The read exists to make a retirement recoverable. If it could raise it would take down the
    session it is only meant to observe — a recovery mechanism that breaks the thing it records is
    worse than none. A failed read costs the detail, not the session."""
    import asyncio as aio

    r, rows, dropped = _retire_fixture(account={"RBRK": 0}, claims={"RBRK": 14},
                                       attributed={}, load_raises=True)
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))          # must not raise
    assert "RBRK" in dropped, "the retirement itself was prevented by its own audit trail"
    released = [d for s, d in rows if "released" in s]
    assert released and released[0]["released"] == ["RBRK"], (
        "the symbol names must survive even when the trail read fails")


def test_a_POSITIVE_account_quantity_retires_nothing():
    """The control. CGAU and HALO are net-positive on paper right now and must be untouched — a fix
    that retired everything would pass every assertion above."""
    import asyncio as aio

    r, rows, dropped = _retire_fixture(account={"CGAU": 80}, claims={"CGAU": 80},
                                       attributed={"CGAU": 80}, trails={"CGAU": True})
    aio.run(r.run(pd.DataFrame(), "2026-08-04"))
    assert dropped == [], f"retired a claim on a position we hold: {dropped}"


# -- #830: a stale claim on a symbol a SIBLING lane holds ------------------------------------------
# Measured on an Alpaca paper instance 2026-09-09: MOMENTUM-002 claimed LFST 152 while the cache attributed all 154 to
# BCTROT-004 (whose own claim was a stale 2). `absent` needs account ≤ 0 (it is 154); `disowned`
# needs the sibling's CLAIM ≥ account (2 < 154). Kept forever, and the sizer refused to size an exit
# off it every slot. The sizer reads ATTRIBUTION as evidence of zero; the retirer must read the same.


def test_a_claim_retires_when_attribution_gives_the_whole_position_to_another_lane():
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    # attribution speaks (mine non-empty), is silent about LFST, and hands all 154 to someone else
    retired = retire_claims(owned={"LFST"}, account={"LFST": 154}, mine={"AEM": 10},
                            foreign={"LFST": 2}, theirs={"LFST": 154})
    assert retired == ["LFST"], f"the sibling's attribution accounts for the whole position: {retired}"


def test_a_PARTIAL_sibling_attribution_does_not_retire():
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    assert retire_claims(owned={"LFST"}, account={"LFST": 154}, mine={"AEM": 10},
                         foreign={"LFST": 2}, theirs={"LFST": 100}) == []


def test_silent_attribution_still_keeps_the_claim():
    """`mine` empty = attribution unavailable (a reconciled-in book). `theirs` is then no evidence
    either, whatever it says — the same standard the existing silence rule holds to."""
    from kumo_strategies.strategies.momentum_rotation.runner import retire_claims

    assert retire_claims(owned={"LFST"}, account={"LFST": 154}, mine={},
                         foreign={}, theirs={"LFST": 154}) == []
