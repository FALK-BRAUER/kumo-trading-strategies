# Plan — ETF_AUTO: sector rotation into the leading single name

Status: draft, 2026-08-01. Replaces `.Codex/plans/momentum-rotation-backtest.md` as the active target.
Tracking: [#1](https://github.com/FALK-BRAUER/kumo-trading-strategies/issues/1). Detail lives in the issues; this
document is the overview.

## What we are building

Each morning, find the leading sector or industry ETF. Then trade the strongest **stock inside it** — not
the ETF. The sector picks the direction; the single name is what you buy.

Two clocks:

- **Daily, before the open** — classify liquid sector ETFs, arm a price level on the ones with a setup. A
  setup is a candidate with a level, never a buy.
- **Intraday** — confirm which sector is actually leading, pick the name inside it, wait for the armed level
  to fire on relative volume.

A trade needs three things to agree: weekly permission, a daily arm, an intraday trigger. Armed but not
fired is a watch. Fired without an arm is noise.

Specced in kumo-trading-platform **#16** (sector ETF rip-readiness) → **#45** (leading constituent) → **#37** (entry
quality). The two-clock architecture is **#168**; the general classifier is **#167**.

This is the **top-down** strategy. The bottom-up universe scanner — scan every name, trade whatever
qualifies, cockpit **#9 MOMENTUM** — is a **separate strategy in this repo, later**. PENG ripped 17.7% while
semis fell 5%; this pipeline could not produce it and should not try.

## Why not the monthly rotation plan

The retired plan specified GEM-style monthly evaluation on daily bars. Every cockpit ticket is intraday —
pre-armed buy-stops firing when the regular session clears a level, volume confirmation, holding past the
opening range, momentum-break exits. The origin stories are single sessions.

The monthly **rules** are retired. The **methodology** underneath them is not — see "What we keep".

## Data

**One source for backtest and live: Alpaca SIP** (Algo Trader Plus, active 2026-08-01).

Backtest and live must use the same data. Relative volume is the primary entry filter, so a volume series
that differs between the two silently changes behaviour under identical code.

The 5-minute parquet at `research-lab/data/raw/massive/intraday/` is **retired as a source** — it ends
2026-06-05, its volume is on a different scale, and #37 needs 1-minute bars to detect a 2–5 minute base.
Keep it as an independent cross-check only.

Bars reach the backtest through kumo-trading-platform `backend/scripts/export_bars.py` (raw, unadjusted, manifested).
That exporter has open integrity gaps — cockpit **#183** — which must close before its output is trusted,
since it is now the only path data takes into a backtest.

## Substrate

One `Strategy` subclass under both `BacktestEngine` and the live `TradingNode`. Not two implementations
reconciled by a parity harness — kumo-qc paid that tax repeatedly (#518, #546, #394).

Decision logic stays as **pure functions with no Nautilus imports**, unit-tested standalone; the `Strategy`
subclass only calls them. Fast iteration and one implementation.

Identity: `StrategyConfig(strategy_id="ETF_AUTO", order_id_tag="001")` → `ETF_AUTO-001`.

**The code being identical does not make the environments identical.** Warmup, order lifecycle and clock all
differ between backtest and live, and strategy code can silently depend on the difference — see
[#11](https://github.com/FALK-BRAUER/kumo-trading-strategies/issues/11).

## Three ways this can manufacture edge

Read [#10](https://github.com/FALK-BRAUER/kumo-trading-strategies/issues/10) before writing any classifier. All three
would survive every fixture test and produce a backtest that looks excellent and is wrong.

1. **The intraday trigger reading its own bar.** A bar's high and volume are only known once it closes, so
   "the level was crossed this bar on above-average volume" is information that did not exist while the bar
   formed. Evaluate on closed bars, fill at the next open.
2. **The pre-open scan using the day it predicts.** In a backtest the current day's daily bar exists, and
   the current week's bar contains the day you are about to trade. Every classifier takes an explicit
   `as_of` and may only read bars completed before it.
3. **Splits.** The data is raw by policy, which is right. But a breakout detector cannot tell a 10:1 split
   from a −90% move — NVDA went ~$1185 → ~$119 in one bar on 2026-06-10. A corporate-action gate is a Phase
   1 blocker.

## What the research changed

A 2024–2026 evidence pass contradicts part of #45's design. Recorded because #45 itself says the weights are
not yet fixed.

- **Premarket sector leadership is a noisy prior.** No published quantification of premarket→intraday
  persistence at the sector level exists. Measure it; confirm leadership 15–30 minutes after the open before
  committing size.
- **#45's primary signal is its weakest.** Premarket relative strength vs the sector has thin evidence.
  Catalyst presence and abnormal premarket volume have strong evidence. The ranking inverts: RS becomes a
  tie-break, and only when already backed by volume and news.
- **Mega-cap domination.** A "sector move" can be one mega-cap while the rest of the sector is flat or
  opposite, so a smaller constituent picked on sector strength is fighting the flows. Guard on
  concentration and breadth.
- **Thin premarket liquidity distorts relative strength.** Use premarket VWAP, not last trade, and impose
  volume and spread minimums.
- **Dispersion is real but conditional.** Names moving 2–4× the ETF's range is close to mechanical. But with
  no selection edge, single names add uncompensated idiosyncratic risk and the ETF is strictly better with
  more size. The name layer must justify itself — hence the baseline test in Phase 5.

The evidence behind this section came from a **search summary, not primary reads**, and is weaker than the
momentum note it replaces. Provisional until [#8](https://github.com/FALK-BRAUER/kumo-trading-strategies/issues/8).

## Phases

| | | |
|---|---|---|
| 1 | Data in — ingest integrity + fill model, as two separate gates | #2 |
| 2 | Sector layer — weekly permission, daily arm, intraday leadership confirm | #3 |
| 3 | Name layer — constituent selection inside the leading sector | #4 |
| 4 | Entry quality — readiness score, anti-chase veto, vol-adaptive sizing | #5 |
| 4.5 | Evidence harness — costs, metrics, walk-forward, PBO | #12 |
| 5 | Does it pay, and does the name beat the ETF? | #6 |

Phase 5 is allowed to kill the design and should be run as if that is the expected outcome. Its central
question is whether the selected constituent beats a **random** constituent of the same leading sector — if
not, the ranking is decoration and the strategy should trade the ETF.

## Package layout

This repo will hold several strategies. Anything a second strategy would plausibly want — Ichimoku, ADX,
relative volume, bar loading, fill and cost models, evidence artifacts — must not live inside the first
strategy's folder. See [#9](https://github.com/FALK-BRAUER/kumo-trading-strategies/issues/9).

## What we keep from the old literature work

`literature.md` read three papers in full and is explicit about which sources it
could not access. Its trading rules are wrong for us; its discipline is not, and should be extracted into a
standing standards document ([#7](https://github.com/FALK-BRAUER/kumo-trading-strategies/issues/7)):

- Do not conclude that because an effect exists, a naive version of it works. The supported system is
  always signal *plus* risk gate *plus* cost control.
- Do not tune the high-leverage knobs until results look good. The first pass should be deliberately boring.
- Do not treat a published performance figure as evidence until you can reproduce its data basis.
- Raw executable prices only. Dividends are ledger entries, never price adjustments. Splits are a separate
  data-quality gate.
- Costs and turnover are first-class — more so intraday, which trades far more.
- Volatility-scaled sizing. Intraday form: widen the stop for a volatile name and cut size to fit; never
  tighten the stop to fit the size. PENG was exactly this failure.
- Report gross and net separately, plus turnover and average holding time.
- Declare which sources you actually read and which you could not.
