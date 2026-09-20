"""TEMPLATE session gateway — copy this alongside the adapter to wire a new lane.

THE ORDER OF OPERATIONS IS THE SAFETY MODEL, so it is fixed here rather than left to callers. Every
`!!` marks a defect that reached production; leave them in your copy until the line each guards has
been deliberately considered.

  1. read the operator's lifecycle. Only a state that MAY act, acts.
  2. decide with the PURE functions the backtest uses. No second implementation.
  3. size against THIS STRATEGY's holdings and THIS STRATEGY's allocation.
  4. exits before entries.

WHAT THIS DELIBERATELY DOES NOT DO: judge its own health, police its own budget, or infer whether an
order is an entry. `contract.py`: "THE STRATEGY DECLARES, THE PLATFORM DECIDES." A strategy polices
its allocation correctly right up until the day it has a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from kumo_strategies.runtime.executor.broker import Broker, DryRunBroker, OrderRequest
from kumo_strategies.strategies.template import (
    TemplateConfig, build_feature_panel, decide, rebalance_dates)


@dataclass
class TemplateSessionRunner:
    """One session. Satisfies the shape `dry_run()` and the adapters call: `run(panel, session)`."""

    cfg: TemplateConfig
    broker: Broker = field(default_factory=DryRunBroker)
    strategy_id: str = "TEMPLATE-000"
    #: !! THIS STRATEGY's capital, never the account's. Sizing off account equity is how a 20k
    #: strategy on a 100k account builds two oversized positions where the research measured ten.
    allocated_equity: float | None = None

    async def run(self, panel: pd.DataFrame, session: str, jobs: object | None = None,
                  slot: str = "open+5m", opens: dict | None = None):
        """`jobs`, `slot` and `opens` are accepted by EVERY runner even where unused — the
        SessionRunner protocol. `opens` joins them (2026-09-05): the gap dead band needs TODAY's open, which the
        panel cannot carry because it is trimmed to sessions strictly before the one being decided.
        Unused here, accepted because the protocol is "every runner takes every shape".

         On 2026-08-17 an adapter began passing `slot=` and its paired gateway did not accept
        it: every session died on the call with `TypeError: run() got an unexpected keyword argument`,
        nothing journalled it, and two trading days were lost behind 41 and 75 rows of ordinary
        chatter. Accepting the known superset means an adapter can pass either without knowing which
        runner it is wired to.

        NOT `**kwargs`. Tolerating anything would swallow a typo — `slott=` would be silently dropped
        and the behaviour it asked for would never happen. The superset is explicit so an UNKNOWN
        kwarg still fails loudly, and `test_session_runner_protocol` catches a new one at the seam.

        `jobs` is momentum-specific (pool refresh) and unused here.
        """
        if panel is None or panel.empty:
            return None
        day_ts = pd.Timestamp(session)
        # !! Cadence from the CONFIG, via the PURE function the backtest uses. A second adapter-local
        # notion of "first session of the month" is a second derivation, and two derivations disagree.
        if day_ts not in set(rebalance_dates(panel["date"], self.cfg.rebalance_period)):
            return None

        scored = build_feature_panel(panel, self.cfg)
        day = scored.loc[scored["date"] == day_ts]
        if day.empty:
            return None

        # !! WHAT THIS STRATEGY OWNS — never `broker.positions()`, which is the ACCOUNT's book.
        # TECHIVOL-005 read the account and formed eight liquidation orders against two other
        # strategies' positions on its first live session.
        held = set(self._own_positions())
        d = decide(day, self.cfg, held)

        # !! EXITS FIRST. Selling before buying frees cash and the venue's share reservation before
        # anything asks for them, so a rotation cannot be half-applied into a shortfall.
        for sym in sorted(d.exit):
            qty = int(abs(self._own_positions().get(sym, 0)))
            if qty > 0:
                self.broker.submit(OrderRequest(symbol=sym, side="SELL", qty=qty, session=session,
                                                strategy_id=self.strategy_id))

        equity = self.allocated_equity if self.allocated_equity is not None else self._equity()
        # !! A missing account frame is None, not zero — and a lane must degrade to INACTION rather
        # than to an exception at boot. Both of QC345's live failures were equity reading as None.
        if not equity or equity <= 0 or not d.enter:
            return None
        slot = equity / max(self.cfg.portfolio_size, 1)
        for sym in d.enter:
            px = self._price(sym)
            # !! No stale fallback for sizing. Yesterday's close is wrong by exactly the overnight gap
            # and the position cap counts positions, not dollars, so nothing downstream notices.
            if not px or px <= 0:
                continue
            qty = int(slot // px)
            # !! Never submit a zero-quantity order. "sizing yielded 0 shares" was QC345's second
            # live failure and it submitted nothing while looking healthy.
            if qty > 0:
                self.broker.submit(OrderRequest(symbol=sym, side="BUY", qty=qty, session=session,
                                                strategy_id=self.strategy_id))
        return None

    # -- broker reads, each narrowed to THIS strategy ------------------------------------------
    def _own_positions(self) -> dict:
        fn = getattr(self.broker, "strategy_positions", None)
        return dict(fn()) if fn is not None else {}

    def _equity(self):
        try:
            return self.broker.equity()
        except Exception:                                              # noqa: BLE001
            return None

    def _price(self, symbol: str):
        fn = getattr(self.broker, "last_price", None)
        return fn(symbol) if fn is not None else None
