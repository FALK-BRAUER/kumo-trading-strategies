# executor

Runtime that makes the momentum rotation operable: layered symbol pool, lifecycle with a SHADOW
dry-run state, append-only action log, the `Broker` seam, HTTP API and an operator UI.

Venue access goes through Nautilus (#75). `AlpacaBroker` lived in `broker.py` and is deleted, so
there is no direct-to-venue ORDER path here.

`tradable.py` still calls Alpaca REST directly and is the exception, not the rule — it resolves
symbol -> listing exchange, blocks ibkr-paper-retired from booting without an Alpaca key, and is tracked in
#75 phase 2. `tests/runtime/test_no_direct_venue_access.py` records it so the list cannot grow.

Implements kumo-trading-platform issues #185–#193 standalone; the seams match that spec so it merges into the
cockpit behind the same API surface.

## Storage — Postgres

App state lives in **Postgres**, per the project's storage architecture (Nautilus owns trade state,
Parquet holds bars, Postgres holds app data). Tables are namespaced `exec_` and sit alongside the
cockpit's own (`watchlist_item`, `command_ledger`, `trade_cycle`, `manager`).

| module | role |
|---|---|
| `store.py` | SQLAlchemy models + engine. `KUMO_DATABASE_URL`, the same env the cockpit uses |
| `pgpool.py` | symbol pool — sources replaced wholesale, operator whitelist/blacklist above them |
| `pgjournal.py` | action log — append-only, decision rows carry their basis |
| `lifecycle.py` | DISABLED → WARMUP → SHADOW → TRADING; halts automatic, starts operator-only |
| `sources/` | pluggable pool contributors (`ledger_book`, `static_list`) |
| `jobs.py` | per-source refresh cadence + session gating |
| `portfolio.py` | pool / target / actual, and the drift between them |
| `broker.py` | the `Broker` protocol + `OrderRequest`/`OrderResult`; `DryRunBroker` is the default. No venue access — orders go through `runtime.nautilus.broker.NautilusBroker` |
| `runner.py` | `RiskLimits`, `SessionResult`, the session-runner shape |
| `daily_loss.py` | the one daily-loss halt every session runner calls |
| `pgrunner.py` · `qc27_runner.py` · `template_runner.py` | re-export SHIMS (ks#211). The session runners live with their strategies: `strategies/momentum_rotation/runner.py` (`PgSessionRunner`, MOMENTUM and BCTROT), `strategies/qc27_tech_inverse_vol/runner.py`, `strategies/template/runner.py` |
| `api.py` · `web/` | HTTP API + operator UI |

`pool.py` and `journal.py` are the earlier local-file implementations, kept only for the offline
tests. **Postgres is the real store** — an earlier SQLite pass cost two bugs that do not exist in a
pooled client/server database (connections are thread-affine, and two concurrent reads on one
connection raise "bad parameter or other API misuse"). Verified: 20 concurrent reads, 0 failures.

## Invariants worth not breaking

- **The pool governs buying only.** A held name whose source drops it stays under the exit rule.
  Only an explicit operator blacklist forces a sale.
- **Only an operator may enter TRADING.** Anything may halt it; nothing may start it.
- **A source that cannot fetch RAISES.** Returning an empty set would silently empty the pool.
- **A session decides once.** The journal is the idempotency key.

## What runs where

**The strategy runs in the engine node, not here.** The session is fired by the strategy's own
Nautilus clock (`strategies/momentum_rotation/nautilus.py`), market data comes off the node's data
client, and orders go through its exec client. Cockpit wires it up in
`backend/strategies/momentum.py`, behind `KUMO_MOMENTUM_ENABLED`.

**This package is the operator surface plus the state the node has no opinion about**: the symbol
pool, the lifecycle, the action log, the durable give-back trail.

```bash
export KUMO_DATABASE_URL=postgresql+asyncpg://kumo:kumo@localhost:5432/kumo
python -m kumo_strategies.runtime.executor.serve
```

It observes and steers. It does not decide and it cannot trade — a `--schedule` flag and a
`POST /api/run` used to do both, and both are gone: the first was a second clock racing the node's,
the second submitted orders the node knew nothing about while consuming the session's idempotency
key, which would then block the real session that day.

Running it is optional; the strategy trades whether or not this is up.

The compose Postgres has no host port mapping; reach it with a socat sidecar on the compose network
(`docker run --network kumo-paper_bus -p 15432:5432 alpine/socat tcp-listen:5432,fork
tcp-connect:postgres:5432`) or run this inside compose — which is where the node already is.

Does NOT hold: strategy rules (→ `strategies/`), the Nautilus adapter (→ `runtime/nautilus/`),
backtesting (→ `backtesting/`).
