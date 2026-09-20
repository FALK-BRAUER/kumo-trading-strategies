"""Configuration for SMHGLD — a fixed-weight sleeve of semis and gold.

THIS LANE DELIBERATELY HAS NO SIGNAL, and that absence is its main finding rather than an omission.
It replaces a trend-switching version of the same pair. Eleven timing mechanisms were tested on this
exact pair over 2021-2026 and 2024-2026 — binary trend switches at five lookbacks, continuous tilts
toward and against the trend, volatility targeting at three windows, and rotation into four
different defensive assets. Under a null that re-runs the lookback sweep inside every permutation,
none separated from chance (p = 0.25 to 1.00). The binary switch this lane replaces drew down 36.4%,
WORSE than holding SMH outright, because it sells after falls and buys back after rallies.

Tilting in either direction made the drawdown worse, monotonically with tilt size. That is the
result that closed the question: if leaning with the trend and leaning against it both degrade the
book, there is no timing information in the series, only the cost of acting on one.

What does work is the weight. At 48% SMH the sleeve returned 46.0% CAGR with a 14.8% maximum
drawdown over 2024-2026, against 57.0% and 35.7% for SMH alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kumo_strategies.strategies.market_view import MarketAction, MarketSignal, MarketViewConfig

#: A live Nautilus `Bar` carries OHLCV and NOTHING ELSE. `close_adj` cannot come off the wire, so a
#: config defaulting to it backtests fine and raises at its FIRST REBALANCE, weeks later.
PriceField = Literal["close", "close_adj"]
RebalancePeriod = Literal["M", "W", "D"]
#: Where in the session the rebalance fills. Only `close` has been measured.
FillSlot = Literal["close", "open"]


@dataclass(frozen=True)
class SmhGldSleeveConfig:
    """FROZEN, so a runtime layer cannot quietly mutate what the backtest validated."""

    #: The growth leg.
    risk_asset: str = "SMH"

    #: The diversifying leg. NOT cash, and NOT safe: gold's own maximum drawdown over 2024-2026 was
    #: 26.4%, deeper than the sleeve's. It earns its place by being weakly correlated (+0.21), not
    #: by being stable, and in 2022 it returned -0.8% while semis fell 33.5% — it did not rescue the
    #: book that year, it merely declined to participate.
    defensive_asset: str = "GLD"

    #: Fraction of the sleeve in the risk leg. 48%, BY DECISION (2026-09-13; #219), with the
    #: cost of that decision measured by the lab (research-lab research/gld-hedge/bars_long.json,
    #: 0.25pp drift band, 2 bps on traded delta, fixed-weight walk, reproduced 2026-09-13):
    #:
    #:     window               weight   CAGR    Sharpe  maxDD    2022     trades/yr
    #:     2018-01 -> 2026-09   0.32     +21.8%  1.23    -23.0%   -11.0%   146
    #:                          0.48     +25.2%  1.21    -27.6%   -16.3%   156
    #:     2005-01 -> 2026-09   0.32     +14.7%  0.92    -32.4%   -11.0%   132
    #:                          0.48     +16.1%  0.93    -37.3%   -16.3%   143
    #:     2024-01 -> 2026-09   0.32     +41.9%  1.76    -15.7%            151
    #:                          0.48     +46.4%  1.74    -14.8%            160
    #:
    #: Sharpe is flat. The trade is +3.4pp of CAGR (2018-) / +1.4pp (2005-) / +4.5pp (2024-) for a
    #: drawdown 4.6-4.9pp deeper and a 2022 5.3pp worse. That is the price, and it was accepted.
    #:
    #: THIS WAS 32%, chosen against the ulcer index on 2021-2026 ("the objective is a book that can
    #: be HELD through a decline"). The measurement behind that choice still holds — 32% is the
    #: shallower book on every window that contains 2022 — and the decision overrode it for
    #: return. Size the sleeve for a ~28% drawdown on a 2022, not the 23% the old weight carried.
    #:
    #: EVERY PLACE THAT SAYS THE SPLIT DERIVES IT FROM HERE (`weight_split`): the runtime label,
    #: the live notes. Three copies of "48/52" existed before #219 and two disagreed with this
    #: field (#186); a copy that agrees by coincidence is the one nobody checks.
    risk_weight: float = 0.48

    #: Trade only when the risk leg drifts this far from `risk_weight`. A band trades on how far
    #: the book has actually moved rather than on the calendar, which is the cheaper instrument for
    #: the same job.
    #:
    #: A QUARTER of a point, swept from 0.25 to 10 over both windows at the EARLIER 32% weight,
    #: where the tightest band tested was best on every risk measure at once — drawdown 23.1%,
    #: ulcer 6.23%, 2022 at -10.9% — and best on return too; the lab's 48% walk (#219) keeps the
    #: same 0.25pp band. Widening it only helps beyond about 8 points, and
    #: then only by trading so rarely that the sleeve drifts into semis during a rally and carries
    #: that exposure into the fall.
    #:
    #: Cost does not argue the other way, because a tight band makes each trade correspondingly
    #: SMALL: 0.75% of the sleeve on average, against 5% at a 3-point band. At 50bp per unit traded,
    #: five times the assumption used here, the quarter-point band gives up 0.4 points of CAGR.
    #:
    #: This was 5 points, then 3. Both were fairly called inert at 3-9 trades a year. At a quarter
    #: point the lane rebalances about 93 times a year, roughly twice a week.
    drift_band: float = 0.0025

    #: `close` for anything live. See PriceField.
    price_field: PriceField = "close"

    #: How often the drift is CHECKED. Distinct from how often it trades, which the band decides.
    #: Cadence lives in the CONFIG so research and production cannot disagree about it: QC27's
    #: cadence was a runner argument while both production call sites were hardwired monthly, so a
    #: sweep could report daily and the live lane still rebalanced monthly, silently.
    rebalance_period: RebalancePeriod = "D"

    #: The lane's declared view of its own market: its equal-weight SMH+GLD index against its
    #: 50-session average, EXIT_ONLY on risk-off. MEASURED FOR THIS LANE (research-lab
    #: `research/workshop/smhgld_market_view.py`, one-day shift, fill at close, 2 bps, drift-band
    #: rebalance as deployed; risk-off 28% of sessions 2018-26):
    #:
    #:     action on risk-off      2018-26 CAGR / Sharpe / maxDD   2022     2005-26 CAGR / Sharpe / maxDD
    #:     none (before #212)      22.1% / 1.25 / -23%             -11.0%   14.7% / 0.93 / -32%
    #:     EXIT_ONLY               identical -- both legs always held; rebalance sells AND buys continue
    #:     LIQUIDATE (emergency)   14.2% / 1.04 / -18%             -4.2%     7.4% / 0.65 / -28%
    #:     flatten SMH leg only    18.9% / 1.24 / -18%             -6.8%    11.3% / 0.81 / -33%  (not in protocol)
    #:
    #: THIS USED TO DEFAULT TO signal=NONE and call it a measurement: "eleven timing mechanisms were
    #: tested on this pair and none separated from chance, so there is no window this lane could
    #: block on." That conflated two things. The #177 finding is that a timing signal INSIDE the
    #: lane carries no information. The platform view is a GOVERNANCE contract: cockpit must be able
    #: to read this lane's state and, in an emergency, act on it, and a lane answering NONE is
    #: invisible to the poller (#212; 2026-09-13: "no strategy can opt out").
    #:
    #: EXIT_ONLY is a no-op for REBALANCING — this sleeve never opens a name it does not hold, so
    #: blocking entries changes no order (pinned: identical orders with and without the view on a
    #: risk-off day). It is not a no-op for GOVERNANCE. LIQUIDATE is the cockpit's emergency act,
    #: priced above at -8pp CAGR for -5pp drawdown — never this lane's decision.
    market_view: MarketViewConfig = field(default_factory=lambda: MarketViewConfig(
        signal=MarketSignal.INDEX_VS_MA, window=50, action=MarketAction.EXIT_ONLY))

    #: The session slot the rebalance is filled in. MARKET-ON-CLOSE, stated because it was assumed
    #: and unwritten for the whole of this lane's research — the same shape as a `fill_hour` default
    #: nobody chose, which turned a 31% measurement into 7% elsewhere in this repo.
    #:
    #: It is load-bearing for one reason: the drift that triggers a rebalance is computed from the
    #: PRIOR close, and the trade is sized on the close it fills at. Those are the same instant for
    #: a market-on-close fill and are NOT the same instant at an open fill, where sizing on the
    #: close would be look-ahead.
    fill_slot: FillSlot = "close"

    #: There is no trend window to warm up, so this only has to be long enough to establish that
    #: both feeds are alive and printing -- the sleeve genuinely can trade on its second bar.
    #: It was 20 for no reason, which held the lane in cash through January 2024 and cost 2.1
    #: points of measured CAGR against the research. A warmup that is not required by a feature is
    #: not caution, it is an unfunded position in cash.
    warmup_sessions: int = 3

    #: Sessions of missing data before this lane refuses to decide rather than deciding on an old
    #: panel. 4, matching `momentum_rotation`'s `max_stale_days`, and it is REQUIRED rather than
    #: cautious here (#207).
    #:
    #: Before #207 a stale feed produced an EMPTY day-slice and the lane held with a named reason --
    #: loud, and safe. #207 appends a placeholder row for the session so `asof_close` resolves to the
    #: last completed close, which means the slice ALWAYS finds rows and THAT LOUD SIGNAL IS GONE.
    #: Without this threshold the lane would decide on whatever the newest real bar happens to be,
    #: silently, which is the exact failure class #205 removed -- re-entering through the fix for it.
    #:
    #: A stale decision is not a cautious one: it trades last week's prices at today's.
    max_stale_days: int = 4

    def tag(self) -> str:
        """Artifact identity. Every swept field belongs here or two runs overwrite each other."""
        # 4dp on the band, not 2: the promoted value is 0.0025, which formats to "0.00" at two
        # decimals and collides with a band of zero — a different lane writing to the same artifact.
        return (f"{self.risk_asset}-{self.defensive_asset}__w-{self.risk_weight:.2f}"
                f"__band-{self.drift_band:.4f}__cad-{self.rebalance_period}"
                f"__px-{self.price_field}__slot-{self.fill_slot}")

    @property
    def universe(self) -> tuple[str, str]:
        return (self.risk_asset, self.defensive_asset)

    @property
    def target_weights(self) -> dict[str, float]:
        """The whole strategy, in one dict. Both legs are held at all times."""
        return {self.risk_asset: self.risk_weight,
                self.defensive_asset: 1.0 - self.risk_weight}

    def __post_init__(self) -> None:
        from kumo_strategies.strategies._validate import check_field_types

        check_field_types(self)
        if self.risk_asset == self.defensive_asset:
            raise ValueError(
                f"risk_asset and defensive_asset are both {self.risk_asset!r}; a sleeve of one "
                f"asset against itself is a single position wearing a diversification label")
        if not 0.0 < self.risk_weight < 1.0:
            raise ValueError(
                f"risk_weight is {self.risk_weight}; a sleeve holds BOTH legs, so a weight of 0 or "
                f"1 means the other leg never trades and the lane is not what its name says")
        if self.fill_slot != "close":
            raise ValueError(
                f"fill_slot is {self.fill_slot!r}; only 'close' has been measured. At an open fill "
                f"the NAV a rebalance is sized on would be a later instant than the fill itself, "
                f"which is look-ahead — measure it before allowing it")
        if not 0.0 <= self.drift_band < min(self.risk_weight, 1.0 - self.risk_weight):
            raise ValueError(
                f"drift_band {self.drift_band} must be non-negative and smaller than the smaller "
                f"leg ({min(self.risk_weight, 1.0 - self.risk_weight):.2f}); a band wider than a "
                f"leg can never trigger, so the sleeve would drift without limit")


def weight_split(cfg: SmhGldSleeveConfig) -> str:
    """The split as a human reads it — "48/52" — DERIVED from `risk_weight`, never typed (#219, #186).

    THE ONE PLACE. The runtime `STRATEGY_LABEL` and `live_notes` both said the split in their own
    words; on 9079067 the label said 48/52 while the config said 0.32, and the notes said 32/68 —
    three copies, two disagreeing. After the flip the label would have agreed by coincidence, which
    is the copy nobody checks. Whole percentages: a label reading "48.0/52.0" is one nobody wrote.
    """
    risk = int(round(cfg.risk_weight * 100))
    return f"{risk}/{100 - risk}"
