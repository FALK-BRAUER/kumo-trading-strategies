# Repository Boundaries

## This repository — the strategies

Executable strategy implementations (`strategies/<name>/`: the pure decision, its config, its Nautilus lane, its
tests), the shared runtime the platform imports (`runtime/`), the one backtest runner (`backtesting/`), and the
evidence for every strategy as backtest SETS (`strategies/<name>/backtests/<set>/`: the data, `run.py`,
`parameters.json`, tracked results). Nothing here calls a broker; the platform does.

## The trading platform

Operator UI, API, command ledger, managed-book surfaces, the exec-client wraps, and the engine-owned projections the
UI displays. It pins this repository and imports `runtime.executor.*` / `runtime.nautilus.*` and each lane.

## The research record (private)

Exploratory studies, sweeps, diagnostics, abandoned hypotheses, pre-registration notes, and the vendor data they
read. Code comments in this repository cite them as `research/<study>/<document>` — those citations are the
provenance of a number or a decision and are kept verbatim; the documents themselves are not published. A study
becomes public when it can run from a set as a variation; until then its conclusion lives in the comment that
cites it and in the strategy's README.

## The Alpaca adapter

A standalone NautilusTrader adapter for Alpaca, its own repository, pinned by the platform like the IBKR adapter.
