# Momentum Rotation Literature Notes

This note records the implementation-relevant takeaways for the first ETF rotation strategy. It separates
source-backed rules from Kumo design choices so the strategy does not inherit unstated assumptions from
research summaries.

## Reading Status

Primary/full-text sources read or extractable:

- Moskowitz, Ooi, and Pedersen, "Time Series Momentum" (JFE 2012).
  https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf
- Asness, Moskowitz, and Pedersen, "Value and Momentum Everywhere" (2008/2009 working paper version).
  https://elmwealth.com/wp-content/uploads/2017/06/valmomeverywhere_asness_moskowitz_and_pedersen__march_2008.pdf
- Daniel and Moskowitz, "Momentum Crashes" (NBER working paper 20439, 2014).
  https://www.nber.org/system/files/working_papers/w20439/w20439.pdf

Useful but weaker sources read:

- Quantpedia asset-class and sector rotation strategy summaries.
  https://quantpedia.com/strategies/asset-class-momentum-rotational-system
  https://quantpedia.com/strategies/sector-momentum-rotational-system
- Optimal Momentum research/FAQ pages for Antonacci-style implementation commentary.
  https://optimalmomentum.com/research-papers/
  https://optimalmomentum.com/faq/
- Robot Wealth and TuringTrader dual-momentum implementation notes.
  https://robotwealth.com/dual-momentum-review/
  https://www.turingtrader.com/portfolios/antonacci-dual-momentum/

Blocked or not yet fully extracted:

- Faber GTAA and relative-strength papers: SSRN/direct links were blocked or moved.
- Antonacci SSRN papers: SSRN security verification blocked extraction.
- Barroso and Santa-Clara, "Momentum Has Its Moments": direct PDF mirror failed extraction; Daniel and
  Moskowitz cite the constant-volatility approach, but do not treat that as a full-paper read.
- Recent 2025/2026 ML/factor-timing papers: useful for research direction, not v1 implementation authority.

## What The Literature Supports

Momentum is an intermediate-horizon return-persistence effect. The strongest implementation precedent is not
short-term chasing; it is 6- to 12-month ranking, commonly with a skip over the most recent month for individual
stocks because of one-month reversal effects.

There are two distinct momentum legs:

- Relative or cross-sectional momentum: rank assets against peers and hold the winners.
- Absolute or time-series momentum: require an asset's own trend to clear a cash/risk-free benchmark before
  taking risk.

ETF rotation is a practical implementation form because it reduces single-name turnover, capacity issues, and
microstructure noise. Monthly rebalance cadence is common in the implementation literature and fits the signal
horizon. Higher-frequency signal checking increases whipsaw risk unless a separate short-horizon model is being
tested.

The crash problem is not a footnote. Daniel and Moskowitz show that momentum crashes are partly forecastable and
cluster in panic states: after market declines, with high volatility, during sharp rebounds. This matters even
for long-only rotation because the strategy can be underinvested or in the wrong defensive asset during a fast
reversal; for long-short momentum it is existential.

Volatility management is a core design component, not a presentation layer. The literature supports scaling or
conditioning momentum exposure by realized/predicted volatility. Kumo v1 should implement a simple realized-vol
target and then test whether it actually reduces drawdown and crash sensitivity on the chosen ETF universe.

Costs and turnover are first-class. Momentum can look robust before costs and become ordinary after costs,
especially when the strategy rotates too frequently or uses too many assets. Trade bands and a small universe are
part of the edge-preservation design, not just operational convenience.

## What Kumo Should Not Infer

Do not infer that "momentum works" means "rank ETFs and buy top 3." The supported system is ranking plus risk
gate plus volatility/cost control plus slow cadence.

Do not compare raw-price backtests directly to total-return paper results. Literature often uses total returns,
excess returns, futures return indexes, or index series. Cockpit-grade execution backtests should use raw
historical executable prices and then model distributions, cash yield, corporate actions, slippage, and fees
explicitly.

Do not optimize the lookback, top-K, universe, and risk-off asset until the strategy looks good. Those are
high-leverage overfit knobs. The first pass should be deliberately boring: fixed universe, fixed 12-1 or
12-month rule depending on asset class choice, fixed K, fixed rebalance calendar, fixed cost assumptions.

Do not treat Antonacci/Faber headline performance figures as Kumo evidence until their exact data basis,
rebalance date, dividend handling, fees, and benchmark definitions are reproduced or intentionally replaced.

## Kumo V1 Constraints

Recommended strategy identity:

- `strategy_id`: `ETF_AUTO-001`
- Cockpit strategy key: `ETF_AUTO` where the UI/detail registry needs the tag-stripped strategy name.
- Model/config provenance: `GEM-VT` or `gem_vt`; this must not replace the Cockpit `strategy_id` field.
- backtest `account_id`: `BACKTEST`
- backtest `client_id`: `SIM`
- `cycle_id`: holding lifecycle per `(account_id, client_id, instrument_id, strategy_id)`, not the whole run id

Recommended raw-data policy:

- Use raw daily OHLCV for signal and execution.
- Label the result as raw-price evidence.
- Add distributions/cash yield later as explicit ledger cash flows, not adjusted prices.
- Treat splits/corporate actions as a separate data-quality gate because raw split-unadjusted series can corrupt
  momentum and execution simulation.

Recommended first rule set:

- Monthly evaluation on the last tradable session after close.
- Small liquid ETF universe.
- Relative momentum rank across risk assets.
- Absolute momentum/cash gate against the safe asset or cash proxy.
- Top-K selection with risk-off fill.
- Realized-vol sizing with no leverage in v1.
- Trade band to suppress small rebalance noise.
- Explicit slippage, fee, and turnover reporting.

Recommended validation:

- Show decisions for 2008, 2020, and 2022.
- Report gross and net metrics separately.
- Report turnover and average holding time.
- Report number of risk-off months and months with partial risk-off fill.
- Compare raw-price result to an adjusted/total-return diagnostic only as a reconciliation exercise, not as the
  production backtest.

## Architecture Implication

`kumo-trading-strategies` should own the pure decision engine and backtest runner. `kumo-trading-platform` connectors can export
historical raw bars into a reproducible data snapshot, but the backtest should run from that snapshot. Cockpit
should later display backtest artifacts and host paper/live execution lanes; it should not become the source of
strategy rules.
