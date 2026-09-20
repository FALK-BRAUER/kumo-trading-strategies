"""The SLEEVE family for the one runner (`engine.SessionEngine`): the SMH/GLD two-leg sleeve — a
fixed-weight book rebalanced at the close on its cadence when the drift leaves the band.

Faithful decomposition of `research/smhgld_sleeve/verify_engine_matches_research.walk` (the
per-strategy walk this replaces, ks#270): mark the sleeve at today's close, hand the strategy's own
`decide` the prior-close features and what THIS sleeve holds, and when it says `opening` or
`rebalance` set both legs to their target weight at that same close, paying the measured cost on
the notional traded. Nothing here decides — the strategy does; this class only expresses the
decision as `Order`s the engine executes.

WHERE IT DIFFERS FROM THE WALK, BY CONSTRUCTION: shares are INTEGERS here (the walk held fractions
of a unit book). At a million dollars the rounding is below a basis point and the research numbers
reproduce inside their stated tolerance; on a 20 k book the rounding alone can exceed the drift
band — a property of a small sleeve, and the set's `book-20k` variation measures it rather than
hiding it behind fractional shares.

NOT MODELLED: `cfg.market_view`. The lane declares a 50-day index-vs-MA view acting EXIT_ONLY; the
walk this reproduces never read it and neither does this family, so the research numbers are the
NO-VIEW numbers. A variation that measures the view is work not yet done, stated here rather than
implied by a field that is read nowhere.
"""
from __future__ import annotations

import pandas as pd

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.engine import Order, SessionEngine
from kumo_strategies.backtesting.sim_venue import Book, LabelDefs, SimVenue
from kumo_strategies.strategies.smhgld_sleeve import SmhGldSleeveConfig
from kumo_strategies.strategies.smhgld_sleeve.engine import build_feature_panel, decide, rebalance_dates

#: The fill instant: the close. Sizing and fill on the same print, as every number on this lane
#: was measured (`live_notes`).
CLOSE = (16, 0)


class SmhGldSleeve:
    def __init__(self, bars: pd.DataFrame, *, cfg: SmhGldSleeveConfig, cost_model: CostModel,
                 starting_cash: float, n_trials: int, benchmarks: dict[str, pd.Series] | None) -> None:
        px = bars.copy()
        px["date"] = pd.to_datetime(px["date"])
        panel = build_feature_panel(px, cfg)
        self.day = {d: frame for d, frame in panel.groupby("date", sort=False)}
        self.close = px[px["ticker"].isin(cfg.universe)].pivot(index="date", columns="ticker", values=cfg.price_field)
        self.sessions = rebalance_dates(panel["date"], cfg.rebalance_period)
        self.cfg = cfg
        self.starting_cash, self.n_trials, self.benchmarks = starting_cash, n_trials, benchmarks
        self.book = Book(starting_cash)
        self.venue = SimVenue(cost_model, LabelDefs("SIM"))
        self.engine = SessionEngine(self.book, self.venue)
        self._decisions: list[dict] = []
        self._target: dict[str, int] = {}
        self.rebalances = 0
        self.unknown_sessions = 0

    # -- the engine's hooks ------------------------------------------------------------------------

    def _ts(self, d) -> pd.Timestamp:
        return pd.Timestamp(d) + pd.Timedelta(hours=CLOSE[0], minutes=CLOSE[1])

    def _prices(self, d) -> dict[str, float]:
        return {s: float(self.close.loc[d, s]) for s in self.cfg.universe}

    def skip(self, d) -> bool:
        day = self.day.get(d)
        return day is None or day.empty or any(pd.isna(self.close.loc[d, s]) for s in self.cfg.universe)

    def session_open(self, d) -> list[Order]:
        return []

    def slots(self, d) -> list:
        return [("close", CLOSE)]

    def before_decision(self, d, slot) -> None:
        return None

    def decide(self, d, slot) -> bool:
        cfg, book = self.cfg, self.book
        price = self._prices(d)
        decision = decide(self.day[d], cfg, set(book.held), {s: float(q) for s, q in book.held.items()})
        self._decisions.append({"date": pd.Timestamp(d), "regime": decision.regime,
                                "weights": dict(decision.weights), "reasons": list(decision.reasons)})
        if decision.regime == "unknown":
            self.unknown_sessions += 1
        if decision.regime not in {"opening", "rebalance"}:
            return False
        # The sleeve's value at this close, before the cost of moving it — the walk's `equity`.
        equity = book.cash + sum(q * price[s] for s, q in book.held.items())
        self._target = {s: int(decision.weights[s] * equity / price[s]) for s in cfg.universe}
        self.rebalances += 1
        return True

    def exit_orders(self, d, slot) -> list[Order]:
        # Sells first, so the buys are sized against the cash they free.
        ts, price, out = self._ts(d), self._prices(d), []
        for s in sorted(self.cfg.universe):
            have, want = self.book.held.get(s, 0), self._target[s]
            if want < have:
                self.book.held[s] = want
                if want == 0:
                    self.book.held.pop(s)
                out.append(Order(s, "SELL", have - want, price[s], ts, "set", at=CLOSE))
        return out

    def entry_orders(self, d, slot) -> list[Order]:
        ts, price, out = self._ts(d), self._prices(d), []
        for s in sorted(self.cfg.universe):
            have, want = self.book.held.get(s, 0), self._target[s]
            if want > have:
                self.book.held[s] = want
                out.append(Order(s, "BUY", want - have, price[s], ts, "set", at=CLOSE))
        return out

    def session_close(self, d) -> list[Order]:
        return []

    def after_close(self, d) -> None:
        return None

    def marks(self, d) -> dict[str, float]:
        return self._prices(d)

    def curve_row(self, d, value: float) -> bool:
        return True

    def decisions(self) -> list[dict]:
        return self._decisions

    def diagnostics(self) -> dict:
        return {"rebalances": self.rebalances, "unknown_sessions": self.unknown_sessions,
                "sessions": len(self.sessions)}
