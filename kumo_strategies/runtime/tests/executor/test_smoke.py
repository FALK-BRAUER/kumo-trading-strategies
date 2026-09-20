"""The startup dry run: prove a lane can act BEFORE it is live (2026-08-22).

"no shadow shit. test, dryrun etc. but shadow is useless. a strategy crashing in shadow or live
doesn't make a difference."

That is right and it is why SHADOW was dropped as the answer. SHADOW runs the SAME code down the SAME
path — it does not prevent a defect, it only delays discovery by one promotion, and only if somebody
reads the journal. TECHIVOL-005 would have formed the same eight liquidation orders in SHADOW; the
only difference is they would not have been sent, which is luck expressed as policy.

A dry run is different in kind: KNOWN inputs, a CONTROLLED broker, and an assertion about the orders
that come out. It fails at startup, deterministically, before any market data exists — and it fails
the same way on a Sunday as on a Monday.

WHAT THIS CATCHES THAT PREFLIGHT DOES NOT. Preflight (kumo-trading-platform issue 438) asks "can I read equity,
positions, prices". This asks "given a book, do you form the RIGHT orders" — which is where the
ownership bug and the position-cap bug live, and neither is visible to a capability probe.
"""

from __future__ import annotations

import pandas as pd


def _panel(symbols, sessions=140):
    dates = pd.bdate_range("2026-01-01", periods=sessions)
    rows = []
    for i, sym in enumerate(symbols):
        base = 50.0 + i
        for j, d in enumerate(dates):
            px = base * (1.0 + 0.004 * j + 0.01 * ((i + j) % 5) / 5)
            rows.append({"ticker": sym, "date": d, "open": px, "high": px * 1.01,
                         "low": px * 0.99, "close": px, "volume": 5_000_000.0})
    return pd.DataFrame(rows)


def test_a_lane_that_forms_no_order_at_all_is_reported():
    """QC345's signature, twice live: it decided five entries and formed nothing. A dry run that
    cannot tell "correctly did nothing" from "structurally incapable of acting" is worthless, so the
    scenario is built to REQUIRE action — an empty book, funded, with prices."""
    from kumo_strategies.runtime.executor.smoke import dry_run

    findings = {f.name: f for f in dry_run(_inert_gateway(), _panel([f"S{i}" for i in range(12)]))}
    assert findings["orders_formed"].value == 0
    assert findings["orders_formed"].error is None, "an inert lane is an observation, not a crash"


def test_an_order_against_a_FOREIGN_position_is_reported():
    """TECHIVOL-005, 2026-08-21: eight liquidation orders against BCTROT's and MOMENTUM's book. The
    dry run seeds foreign holdings the lane does not own and reports any order touching them."""
    from kumo_strategies.runtime.executor.smoke import dry_run

    findings = {f.name: f for f in dry_run(_foreign_selling_gateway(),
                                           _panel([f"S{i}" for i in range(12)]),
                                           foreign={"AEM": 18, "BDX": 65})}
    assert set(findings["foreign_orders"].value) == {"AEM", "BDX"}, (
        "a lane forming orders against another strategy's book was not reported")


def test_a_healthy_lane_reports_orders_and_no_foreign_touches():
    from kumo_strategies.runtime.executor.smoke import dry_run

    findings = {f.name: f for f in dry_run(_healthy_gateway(), _panel([f"S{i}" for i in range(12)]),
                                           foreign={"AEM": 18})}
    assert findings["orders_formed"].value > 0
    assert findings["foreign_orders"].value == []


def test_a_gateway_that_raises_is_REPORTED_not_propagated():
    """A dry run that throws tells you less than one that reports — and it must never take the node
    down at boot, which is the failure mode `build_optional_strategy` already guards for."""
    from kumo_strategies.runtime.executor.smoke import dry_run

    findings = {f.name: f for f in dry_run(_exploding_gateway(), _panel(["S0"]))}
    assert findings["ran"].error is not None
    assert "boom" in findings["ran"].error


# -- gateways under test ---------------------------------------------------------------------------
class _Base:
    def __init__(self):
        self.broker = None

    async def run(self, panel, session, jobs=None, slot=None, opens=None):
        return None


def _inert_gateway():
    return _Base()


def _foreign_selling_gateway():
    class G(_Base):
        async def run(self, panel, session, jobs=None, slot=None, opens=None):
            from kumo_strategies.runtime.executor.broker import OrderRequest
            for sym in self.broker.positions():                 # THE BUG: the account's book
                self.broker.submit(OrderRequest(symbol=sym, side="SELL", qty=1, session=session,
                                                strategy_id="SMOKE-000"))
    return G()


def _healthy_gateway():
    class G(_Base):
        async def run(self, panel, session, jobs=None, slot=None, opens=None):
            from kumo_strategies.runtime.executor.broker import OrderRequest
            for sym in sorted(panel["ticker"].unique())[:3]:
                self.broker.submit(OrderRequest(symbol=sym, side="BUY", qty=10, session=session,
                                                strategy_id="SMOKE-000"))
    return G()


def _exploding_gateway():
    class G(_Base):
        async def run(self, panel, session, jobs=None, slot=None, opens=None):
            raise RuntimeError("boom")
    return G()
