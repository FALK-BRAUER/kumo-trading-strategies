# One runner architecture — the backtest executes the live runner's phases (#270)

Design decision, 2026-09-18: *"make a good unified runner architecture that is inline with the actual system
runner (maybe adds features that are not yet in system runner)."* The system runner is the live
executor, `strategies/momentum_rotation/runner.py` (`PgSessionRunner.run`, one call per decision slot;
until ks#211 this file was `runtime/executor/pgrunner.py`, and that path is now a re-export shim). This
document sets it beside the backtest runner as it stands after folds 1–3 and states the target.
Read at main `47c8033`.

## 1. The live runner's phases, and who owns each today

| # | phase (one slot of one session) | owner today | notes |
|---|---|---|---|
| L1 | lifecycle gate — TRADING / PAUSED / LIQUIDATING, urgent exits first | `lifecycle.State`, `pgrunner.run:715–760` | `st.decides`, `st.may_submit_entries/exits` |
| L2 | feed health + daily-loss halt | `pool.sources()`, `daily_loss.enforce` (`:756`) | BLOCK / HALT verdicts, anchored equity |
| L3 | ownership — account vs claims, orphans adopted, stale claims retired, foreign/short positions journalled, **held ≠ sellable** | `_owned`, `_foreign_claims`, `retire_claims`, `reducible`, `own_ceiling` (`:58–398`, `:767–1020`) | cockpit's exec client adds a second gate (`exit_ownership.classify_exit`) |
| L4 | decided-already / idempotency per (session, slot) | `journal.decided_this_session` (`:1031`) | duplicate `client_order_id` is load-bearing |
| L5 | panel → gates → score | `apply_gates`, `score_panel` (`engine.py`), `:1073` | same two calls as the backtest |
| L6 | coverage + rankable floors | `limits.min_bar_coverage`, `min_rankable_frac` (`:1094–1130`) | refuse thin sessions |
| L7 | trail exits (give-back, ATR, highs) from persisted trail state | `_trail_exits` → `evaluate_exits` (`exits.py`), `_load_state/_save_state` (Postgres) | ONE evaluator, shared |
| L8 | decision — ranking, corr/vol stats, held-name data gaps | `decide` (`engine.py`), `panel_stats`, `:1160–1180` | same function as the backtest |
| L9 | gap dead band at the fill (last look, live open vs prior close) | `filter_entries_by_gap` (`:1197`) | live knows the open at 09:35 |
| L10 | order plan: exits first, entries wait for unfunded sells, attempts per side | `_submit:1455–1560`, `attempts_for`, `held_sessions_before` | #224 B |
| L11 | sizing to the book: allocated equity × deployed / `sizing_denominator(book_after)`, weight scale, `max_position_notional`, `max_positions` | `_submit:1640–1740`, `RiskLimits` | live sizes off `limits.allocated_equity`, NOT starting cash |
| L12 | submit through the venue port | `broker.submit / broker.exit` (`NautilusBroker`), `OrderRequest` | side derived from `POSITION_SIDE`; cockpit's budget gate and ownership gate sit behind this port |
| L13 | terminal events → claims → journal | `record_terminal`, `sync_claim`, `_resume` | fills move the book, never submit |
| L14 | protection (protective stop on the fill, `release_for_exit` on exit) | cockpit (`release_for_exit`, contract `protective_close`) | NOT in this repo's runner |
| L15 | market-aware view (RISK_ON/OFF → target weights) | `MarketAwareMixin` (`runtime/nautilus/market_aware.py:199`), adapters | above the runner, per lane |

## 2. The backtest runner beside it (`backtesting/runner_sessions.py`, two clocks)

| live phase | intraday path | daily path (former `runner_verified`) | verdict |
|---|---|---|---|
| L5 gates/score | `apply_gates`, `score_panel` — same functions | same | **shared** |
| L7 trail exits | `evaluate_exits` on prices as of the slot; trail in memory | `evaluate_exits` on the close; trail in memory | shared function; **state persistence re-implemented** (dict vs Postgres) |
| L8 decision | `decide`, `panel_stats` — same | same | **shared** |
| L9 gap | `filter_entries_by_gap` at the slot | `gap_declines_entry` at the next open | shared predicate, **two call shapes** |
| L10 order plan | sells first, cash funds buys in the same decision | same | **re-implemented** (`:380–470` and `:620–700`); live's "entries wait for unfunded sells" and attempts are absent |
| L11 sizing | `sizing_denominator(book_after)`, weight scale, rebalance/trim bands | same off `dec.hold` | shared denominator; **`max_position_notional`/`max_positions` absent** |
| L12 venue | fill = next bar after the slot (or the 09:30 print for `moo`) + `CostModel.charge` | fill = next session's open + `CostModel.charge` | **the simulated venue, inline** |
| L1–L4, L6, L13, L14 | — | — | **absent**: no lifecycle, no daily-loss halt, no claims/ownership, no idempotency, no coverage floors, no protection |
| L15 market view | `market_state` → `target_weights_under` (`:300`) | refused | ahead of live's executor (live has it in the adapter) |
| — | `trade_from` warm-up, look-ahead truncation tests, `n_trials` → deflated Sharpe, benchmarks, `Report` | same | **backtest-only, keep** |

Drift measured so far: the six points between the two fill conventions (`runner_sessions` docstring); ks#273 — the engine replay disagrees with the daily path on every fill (35 vs 8, zero shared), which is the shape §3 removes by making the venue the only difference.

## 3. Target: one Session engine, a Venue port, one config surface

```
SessionEngine.run(session, slot)                       # the same object live and in replay
  L1 lifecycle → L2 halt → L3 ownership → L4 idempotency → L5 gates/score → L6 floors
  → L7 trail exits → L8 decide → L9 gap → L10 order plan → L11 sizing → L12 venue.submit
  → L13 terminal events → claims → journal → L14 protection
ports (protocols, one implementation each side):
  Venue     live: NautilusBroker + cockpit's gates             replay: SimVenue(bars, CostModel, fill rule)
            — `budget_allows(order)` and `owns(order)` are PORT METHODS: live delegates to cockpit's
            budget gate and `exit_ownership.classify_exit`; replay implements both against its own
            Book (allocated equity, side-aware ownership), so a budget refusal or an ownership
            refusal is a measurable event in history, not a live-only surprise.
  Book      live: broker.positions + claims (Postgres)         replay: in-memory book + claims
  Journal   live: PgJournal                                    replay: DataFrame journal → Report
  Clock     live: calendar + slot alerts                       replay: bars' session bounds → slot instants
  Trail     live: Postgres trail rows                          replay: dict
```

- **The strategy layer stays pure** (`engine.py`, `exits.py`, `market_view.py`): L5, L7, L8, L9 are already the same functions on both sides — that is the part that must never fork.
- **`SimVenue` is where the eight runners' differences live**, as fill rules selected by the config: market-at-slot → next bar; `moo` → opening print; resting limit (CRSISHORT: `max(limit, open)` if the session's high reaches it, one session); limit-on-open (the live decomposition, #131); short side flips the arithmetic and charges borrow. Costs: `CostModel.charge` at the fill instant. The cost model, look-ahead truncation and deflated Sharpe stay replay-only, on the Journal/Report side.
- **One config surface** — the #270 contract table as fields: `execution.{decision_slots, fill, order_type, min_abs_gap_pct}`, `cadence.{period, forced_dates}` (monthly lanes), `sizing.{mode: slots|fixed|weights, deployed, max_position_notional, max_positions}`, `side` (from the strategy), `exits` (one evaluator per family, dispatched by strategy), `gates.{market_view, borrow, universe}`. Nothing a deployed lane cannot set.
- **What the backtest GAINS from live**: the lifecycle gate (a PAUSED window in history), the daily-loss halt, held-≠-sellable and claims (multi-lane replay on one book), idempotency per slot, coverage floors, protective stops (a stop-out in replay, today invisible), attempts/entries-wait. Each is a live rule that currently cannot be measured; that is the "features not yet in the system runner" direction reversed — first the replay learns live's rules, then new rules are written once.
- **What live GAINS from the backtest**: the market view moves from the adapter into the engine (L15 becomes a phase), the ledger reconciliation invariant CRSISHORT's runner carries (every session: cash + marks = fills ± costs) becomes an engine-wide check, `n_trials` is recorded per lane so paper results carry a deflated Sharpe.

## 4. Fold order, re-derived

1. **Done**: `runner_cadence` (proven equal), `runner_verified` (daily path, gate to the cent), `runner.py` (engine-parity test; finding #273).
2. **Extract `SimVenue` + `Book` from `run_sessions`** (both clocks) with the two fill rules that exist, gate = ledger-book-daily-2024-2026 13.6297/8.8249 and the cadence reference 9.2976 unchanged. This is the refactor every later fold plugs into; without it each fold re-implements L10–L12 a third time.
3. **Monthly pair** (`runner_qc27_verified`, `runner_qc345_verified`): `cadence.period` + forced dates (`rebalance_dates` already shared), `sizing.mode=weights` (target weights, `target_weights_under`), ATR trail between rebalances — gate = each set's recorded numbers.
4. **CRSISHORT**: `side=SHORT`, resting-limit fill rule, borrow gate as a `gates` port, ledger invariant promoted engine-wide — gate = +80.75 % / 231 trades.
5. **QC27-nautilus**: engine-parity case in `engine_parity/replay.py`, delete the runners; #273 is the same question for momentum and is answered first.
6. **Live adopts the engine**: `PgSessionRunner.run` becomes `SessionEngine.run` with the live ports — the day the replay and pgrunner are one call path, drift is impossible rather than tested for. **Gate is a paper measurement, not a test**: for N sessions the engine runs shadow beside `PgSessionRunner` on the SAME live inputs (panel, account, claims, slot) and must produce the IDENTICAL order plan (symbol, side, qty, price) every slot, journalled side by side; one disagreement stops the switch and is a ticket. No cut-over before that read.

Numbers that gate each step are the ones already recorded on #270; a step that cannot reproduce its number stops and explains the delta before deleting anything.
