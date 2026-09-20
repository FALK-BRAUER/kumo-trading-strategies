"""The startup DRY RUN: prove a lane can act before it is live (2026-08-22).

    "no shadow shit. test, dryrun etc. but shadow is useless. a strategy crashing in shadow or live
     doesn't make a difference."

Correct, and it is why SHADOW is not the answer. SHADOW runs the SAME code down the SAME path: it does
not PREVENT a defect, it delays discovery by one promotion, and only if somebody reads the journal.
TECHIVOL-005 would have formed the same eight liquidation orders against BCTROT's and MOMENTUM's book
in SHADOW — the only difference is they would not have been sent, which is luck expressed as policy.

A dry run is different in kind: KNOWN inputs, a CONTROLLED broker, and an assertion about the orders
that come out. It fails at startup, deterministically, before any market data exists, and it fails the
same way on a Sunday as on a Monday.

WHAT THIS CATCHES THAT PREFLIGHT DOES NOT. Preflight (kumo-trading-platform issue 438) asks "can I read equity,
positions, prices" — a capability probe against the real broker. This asks "GIVEN A BOOK, DO YOU FORM
THE RIGHT ORDERS". The position-cap bug and the cross-strategy ownership bug both live here and
neither is visible to a capability probe:

    MOMENTUM-002   held its full book, sold two, and every replacement entry was refused by the names
                   it had just sold. 19 entries lost across seven sessions, each journalled as a tidy
                   "position cap" while the book shrank.
    TECHIVOL-005   owned nothing and formed eight SELLs against two other strategies' positions.

OBSERVATIONS, NOT VERDICTS — the same rule as `Probe` (contract.py) and for the same reason: a lane
that judges its own health reports healthy right up until the day the thing deciding health is the
thing that is broken. `dry_run` returns what it saw; the platform decides what that means.
"""

from __future__ import annotations

import asyncio

import pandas as pd

from kumo_strategies.runtime.executor.broker import DryRunBroker
from kumo_strategies.runtime.nautilus.contract import Probe

DEFAULT_EQUITY = 100_000.0


def dry_run(gateway, panel, *, foreign: dict | None = None, owned: dict | None = None,
            session: str | None = None,
            equity: float | None = DEFAULT_EQUITY) -> list[Probe]:
    """Run ONE session against an honest `DryRunBroker` and report what it formed.

    `foreign` seeds positions belonging to OTHER strategies on the same account. That is the only
    configuration in which a lane reading `positions()` instead of `strategy_positions()` shows up at
    all — with an empty account both reads return the same thing and the bug is invisible, which is
    exactly why every test TECHIVOL had was green.

    Never raises: a gateway that explodes is reported as `ran(error=...)`. This runs at boot, and a
    dry run that took the node down would be a dry run nobody dares wire in.
    """
    # THE LAST SESSION IN THE PANEL, not the first. Running the first guarantees nothing is warm --
    # a cross-sectional lane needs `warmup_sessions` of history before it will rank anything -- so
    # every lane would report zero orders and the dry run would certify nothing while looking green.
    # A default that makes the check vacuous is worse than no default.
    if session is None:
        session = str(pd.Timestamp(panel["date"].max()).date())
    foreign = dict(foreign or {})
    prices = {sym: float(px) for sym, px in
              panel.sort_values("date").groupby("ticker")["close"].last().items()}
    broker = DryRunBroker(starting_equity=equity, prices=prices)
    broker.adopt_foreign(foreign)
    if owned:
        broker.adopt_own(owned)
    gateway.broker = broker

    out: list[Probe] = []
    try:
        asyncio.run(gateway.run(panel, session))
        out.append(Probe("ran", True))
    except Exception as exc:                                       # noqa: BLE001
        out.append(Probe("ran", None, repr(exc)))

    out.append(Probe("orders_formed", len(broker.sent)))
    # ORDERS TOUCHING SOMEONE ELSE'S BOOK. Reported as the symbols themselves rather than a count:
    # "2" sends the reader looking, ["AEM","BDX"] tells them whose.
    out.append(Probe("foreign_orders", sorted({r.symbol for r in broker.sent if r.symbol in foreign})))
    out.append(Probe("zero_qty_orders", sorted({r.symbol for r in broker.sent if r.qty <= 0})))
    # The ORDER of sides, so "exits before entries" is checkable. A rotation that buys
    # before it sells can be half-applied into a cash shortfall.
    out.append(Probe("sides", [r.side for r in broker.sent]))
    out.append(Probe("orders", [(r.side, r.symbol, r.qty) for r in broker.sent]))
    return out
