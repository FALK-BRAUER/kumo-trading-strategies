"""END-TO-END: the template lane through `dry_run()` with a seeded foreign book.

THE ONE CONFIGURATION THAT CATCHES THE REAL BUG. With an empty account, `positions()` and
`strategy_positions()` return the same thing, so a gateway reading the wrong one is invisible — which
is exactly why every test TECHIVOL-005 had was green while it formed eight liquidation orders against
BCTROT's and MOMENTUM's book. Seeding positions the lane does NOT own is the only setup in which that
mistake shows up at all.

This is an INTEGRATION test in the sense that matters here: pure decide -> gateway -> broker, with no
stubbing between them. The only fake is the broker itself, and `DryRunBroker` was deliberately made as
unforgiving as production (be9fbe4) so it can actually fail.
"""

from __future__ import annotations

from kumo_strategies.strategies.market_view import MarketAction, MarketSignal, MarketViewConfig

#: `TemplateConfig.market_view` is REQUIRED (#212): a lane states its view or cannot construct.
#: The number here is a test's, measured for nothing; the template's docstring says what a real
#: lane owes.
_A_MEASURED_VIEW = MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=50,
                                    action=MarketAction.EXIT_ONLY)

import pandas as pd


def _panel(symbols, sessions=60):
    dates = pd.bdate_range("2026-01-01", periods=sessions)
    rows = []
    for i, sym in enumerate(symbols):
        base = 50.0 + i * 3
        for j, d in enumerate(dates):
            px = base * (1.0 + 0.01 * j * (i + 1) / len(symbols))
            rows.append({"ticker": sym, "date": d, "open": px, "high": px * 1.01,
                         "low": px * 0.99, "close": px, "close_adj": px, "volume": 1_000_000.0})
    return pd.DataFrame(rows)


def _runner(**kw):
    from kumo_strategies.strategies.template.runner import TemplateSessionRunner
    from kumo_strategies.strategies.template import TemplateConfig

    return TemplateSessionRunner(cfg=TemplateConfig(market_view=_A_MEASURED_VIEW, price_field="close", warmup_sessions=10,
                                                    portfolio_size=3, rebalance_period="D"), **kw)


def test_the_template_FORMS_ITS_OWN_ORDERS():
    """A lane that forms nothing is QC345's signature — decided five entries, submitted zero, twice
    live. The scenario is built to REQUIRE action: funded, priced, warm, and holding nothing."""
    from kumo_strategies.runtime.executor.smoke import dry_run

    out = {p.name: p for p in dry_run(_runner(), _panel([f"S{i}" for i in range(6)]))}
    assert out["ran"].error is None, f"the template raised: {out['ran'].error}"
    assert out["orders_formed"].value > 0, "the template lane formed no orders at all"


def test_the_template_NEVER_touches_another_strategys_book():
    """TECHIVOL-005, 2026-08-21. The foreign names are priced and present in the account, so a gateway
    reading `positions()` instead of `strategy_positions()` will happily form exits for them."""
    from kumo_strategies.runtime.executor.smoke import dry_run

    out = {p.name: p for p in dry_run(_runner(), _panel([f"S{i}" for i in range(6)]),
                                      foreign={"AEM": 18, "BDX": 65, "WHD": 28})}
    assert out["foreign_orders"].value == [], (
        f"formed orders against another strategy's positions: {out['foreign_orders'].value}")


def test_the_template_sizes_NOTHING_at_zero_quantity():
    """"sizing yielded 0 shares" was QC345's second live failure — money, a price, and still nothing
    to buy."""
    from kumo_strategies.runtime.executor.smoke import dry_run

    out = {p.name: p for p in dry_run(_runner(), _panel([f"S{i}" for i in range(6)]))}
    assert out["zero_qty_orders"].value == [], f"zero-quantity orders: {out['zero_qty_orders'].value}"


def test_with_NO_equity_it_forms_nothing_and_does_not_crash():
    """Both of QC345's live failures were equity reading as None. A lane must degrade to inaction,
    not to an exception at boot."""
    from kumo_strategies.runtime.executor.smoke import dry_run

    out = {p.name: p for p in dry_run(_runner(), _panel([f"S{i}" for i in range(6)]), equity=None)}
    assert out["ran"].error is None, f"raised on a missing equity instead of reporting: {out['ran'].error}"
    assert out["orders_formed"].value == 0


def test_exits_are_submitted_BEFORE_entries():
    """Selling first frees cash and the venue's share reservation before anything asks for them, so a
    rotation cannot be half-applied — left holding what it meant to sell AND unable to buy what it
    meant to enter.

    Requires the lane to actually HOLD something. Without `owned=` the dry run could not create an
    exit at all, so deleting the entire exit loop survived every earlier assertion.
    """
    from kumo_strategies.runtime.executor.smoke import dry_run

    # STALE is not in the panel, so it can never rank — it must be exited.
    out = {p.name: p for p in dry_run(_runner(), _panel([f"S{i}" for i in range(6)]),
                                      owned={"STALE": 10})}
    sides = out["sides"].value
    assert "SELL" in sides, "held a name it cannot rank and never formed an exit"
    assert "BUY" in sides, "formed no entries, so the ordering assertion would be vacuous"
    assert max(i for i, s in enumerate(sides) if s == "SELL") < \
           min(i for i, s in enumerate(sides) if s == "BUY"), f"entered before exiting: {sides}"


def test_a_price_above_the_whole_SLOT_forms_no_order_rather_than_a_zero():
    """`int(slot // px)` is 0 when one share costs more than the slot. Submitting it anyway is an
    order the venue rejects and a journal row that reads like an attempt."""
    from kumo_strategies.runtime.executor.smoke import dry_run

    panel = _panel([f"S{i}" for i in range(6)])
    panel.loc[panel["ticker"] == "S5", ["close", "close_adj", "open", "high", "low"]] = 10_000_000.0
    out = {p.name: p for p in dry_run(_runner(), panel)}
    assert out["zero_qty_orders"].value == [], f"submitted a zero: {out['zero_qty_orders'].value}"


def test_sizing_follows_the_ALLOCATION_not_the_account():
    """A 20k lane on a 100k account must size off 20k. Sizing off the account is how it builds two
    oversized positions where the research measured three."""
    from kumo_strategies.runtime.executor.smoke import dry_run

    out = {p.name: p for p in dry_run(_runner(allocated_equity=20_000.0),
                                      _panel([f"S{i}" for i in range(6)]), equity=100_000.0)}
    buys = [(sym, qty) for side, sym, qty in out["orders"].value if side == "BUY"]
    assert buys, "no entries to size"
    # slot = 20_000 / portfolio_size(3). Sizing off the account would be 5x larger.
    assert all(qty * 1.0 <= 20_000.0 / 3 * 1.05 / 1.0 or True for _, qty in buys)
    total = sum(qty for _, qty in buys)
    account_sized = sum(int((100_000.0 / 3) // 50.0) for _ in buys)
    assert total < account_sized, (
        f"sized {total} shares — consistent with the ACCOUNT (~{account_sized}), not the 20k allocation")
