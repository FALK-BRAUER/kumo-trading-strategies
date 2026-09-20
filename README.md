# kumo-trading-strategies

The strategies that trade on the Kumo platform, as executable code with their evidence beside them: for each
strategy the pure decision (no broker, no clock, no I/O), its configuration, its NautilusTrader lane, its tests,
and the backtest sets that document it — data, parameters and tracked results in one folder. The platform
(`kumo-trading-platform`) pins this package and runs the lanes; nothing here calls a broker.

| strategy | what it does | lane |
|---|---|---|
| `momentum_rotation` | rotates a small book through the strongest names of an externally maintained candidate ledger, one open-anchored decision a session, give-back exits | MOMENTUM |
| `bct` | the same rotation deciding three times a session — a schedule of momentum, not a second strategy | BCTROT |
| `qc27_tech_inverse_vol` | top-ten US technology momentum, inverse-volatility weights, portfolio stop, gold as the cash proxy | TECHIVOL |
| `qc345_rotation` | monthly momentum rotation over a liquidity-ranked universe with a trail between rebalances | QC345 |
| `crsi_short` | mean-reversion short: ConnorsRSI > 90 at the close, a short limit resting one session, reversal/flat exits, borrow-gated | CRSISHORT |
| `smhgld_sleeve` | a two-leg fixed-weight sleeve (semiconductors / gold) rebalanced at the close when the drift leaves its band | SMHGLD |

Layout (`kumo_strategies/`): `strategies/<name>/` one folder per strategy; `runtime/` the shared executor and
Nautilus contract the platform imports; `backtesting/` the ONE runner every strategy's evidence is produced by;
`contracts/`, `evaluation/`, `provenance.py`. `docs/` holds the runner architecture, the lifecycle and the repository
boundaries; `ARCHITECTURE.md` the map.

## Quick start

```bash
uv run --python 3.12 --extra nautilus --extra dev pytest          # the suite, lanes included
python kumo_strategies/strategies/qc27_tech_inverse_vol/backtests/tech-universe-daily-2025-2026/variations/best/run.py
```

Plain `pip` also works: `python3.12 -m venv .venv && . .venv/bin/activate && pip install -e '.[nautilus,dev]'`.
The data each backtest reads sits in its set (`bars.parquet` through Git LFS — `git lfs install` before cloning);
no environment variable, no external download.

## The backtest

Every strategy carries its own backtests beside its code — `kumo_strategies/strategies/<name>/backtests/<set>/`
holds the data the set was run on (`bars.parquet` through LFS plus whatever else the run reads — a sector snapshot,
an anonymised `ledger.csv` where the strategy follows a ledger), `manifest.json` (where the extract came from),
one `report.html` per set and one `variations/<variant>/` per configuration with its `run.py`, `parameters.json`
(every parameter, the strategy config included) and tracked `results/` (factsheet, metrics, fills, trades, equity,
monthly, diagnostics, provenance). Every `run.py` calls the ONE runner with the decision code the lane trades.

| strategy | set | variations | headline (see the set's README for the caveats) |
|---|---|---|---|
| momentum_rotation | `ledger-book-daily-2024-2026` | `best`, `bctrot` | +13.6 % / Sharpe 0.84 · +8.8 % / 0.60 (2024-06 → 2026-09) |
| qc27_tech_inverse_vol | `tech-universe-daily-2025-2026` | `best`, `daily-no-view`, `monthly` | +202 % / 1.71 · +222 % / 1.79 · +193 % / 1.71 (2025 → 2026-08) |
| qc345_rotation | `liquid-top100-daily-2024-2026` | `best`, `cash-buffer-0.75` | +177 % / 1.55 · +122 % / 1.54 (2025 → 2026-08, 2024 warm-up) |
| crsi_short | `short-screen-daily-adjusted-2025-2026` | `best`, `costs-200bps` | +80.8 % / 2.35 / −8.1 % DD · +22.1 % (2025 → 2026-09) |
| smhgld_sleeve | `smh-gld-daily-2021-2026` | `best`, `since-2021`, `book-20k` | CAGR 46 % / 1.74 · 25 % / 1.16 · 46 % on a $20 k book |
| bct | — | is the `bctrot` variation of momentum | |

`python variations/<variant>/run.py` reproduces a variation; the suite reruns every one and compares the tracked
metrics number for number, and checks each set's provenance hashes against its data.

Real Alpaca instruments (true venue and tick size), Alpaca SIP bars, and costs charged per fill from per-symbol
half-spreads measured off SIP NBBO quotes (a flat rate where the set says so).

**Read `deflated_sharpe_prob` before `sharpe`** — it corrects for how many configurations were
tried, and on this research programme that correction is not cosmetic.

Current state and honest caveats: the operating notes kept beside the private manual.

## Disclaimer

This repository is research software. Nothing in it is investment, trading, financial or legal advice, an
offer or a recommendation to buy or sell any security, or a claim that any strategy here is profitable.
The backtest results tracked beside each strategy are simulations on historical data with modelled costs
and fills: they are hypothetical, they can differ materially from what a live account would have earned,
and past performance — simulated or real — does not indicate future results. Trading securities involves
risk, including the loss of the amount invested and, for short positions, losses beyond it.

The software is provided under the LGPL-3.0-or-later **without warranty of any kind** (see `LICENSE`). The
authors are not registered investment advisers or broker-dealers and accept no liability for any decision or
loss arising from the use of this code or these results. If you run it against a brokerage account, you do
so on your own responsibility, after your own review, and in compliance with your own jurisdiction's rules.
Market data referenced here belongs to its providers and is subject to their terms.
