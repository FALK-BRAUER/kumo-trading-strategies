# backtesting

THE ONE RUNNER. `runner_sessions.run_sessions` is the only entry point; the config's TYPE picks the family
(`families/`: rotation — momentum and BCTROT on two clocks; monthly — QC27 and QC345; short — CRSISHORT; sleeve —
SMH/GLD), every family runs through `engine.SessionEngine` with the `sim_venue` Book and Venue ports, and every
strategy's backtest evaluates the same decision code the lane trades. `test_exactly_one_runner.py` refuses a second.

Holds: the runner and its families, the engine and ports, the measured-spread cost model (`costs.py`), real Alpaca
instrument construction (`instruments.py`), KPI and trade-history reporting (`report.py`), the backtest-set layout
(`layout.py`: the eight result files, provenance, factsheet and set report) and `params.py` (a strategy config as
`parameters.json` and back). `tests/engine_parity/` replays a strategy through a Nautilus `BacktestEngine` as the
second derivation of the runner's numbers.

Does NOT hold: strategy rules (→ `strategies/<name>/`), data (→ each set carries its own), evidence (→
`strategies/<name>/backtests/<set>/`), or exploratory studies (private research record).

Run a backtest: `python kumo_strategies/strategies/<name>/backtests/<set>/variations/<variant>/run.py`.
