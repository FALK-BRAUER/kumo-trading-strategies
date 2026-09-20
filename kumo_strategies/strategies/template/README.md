# `_template` — copy this to start a new strategy

Five strategies were each hand-built from whichever predecessor someone happened to read. The result
was four `run()` signatures, three journal schemas, two ownership conventions, two names for equity,
and a contract class defined twice — none of it chosen. kumo-trading-platform measured 19 decided slots since
2026-07-31 with 10 of the 17 live ones bad, and **eight of the ten fell in the three days three
hand-built lanes were added**. MOMENTUM, the oldest, is the only lane that trades reliably.

That is a manufacturing-defect distribution. This directory is the fix.

## What this gives you

Copying it means a new lane **starts conforming** and has to actively break a rule. The deployment
contract in `tests/runtime/nautilus/test_contract.py` DISCOVERS every adapter in
`runtime/nautilus/`, so your copy is judged by the same machinery the live lanes are — from the
first commit, before it has ever seen a market.

## Layout

| file | what it is |
|---|---|
| `config.py` | frozen dataclass. **Every** field a runner or adapter needs lives here |
| `engine.py` | the pure decision. No Nautilus, no broker, no clock, no database |
| `nautilus.py` | the adapter (Nautilus lane) |

## The checklist

1. Copy both directories, rename, set `EXTERNAL_ID` and drop `IS_TEMPLATE`.
2. **Do not choose an `order_id_tag`.** Cockpit allocates it; the constructor must keep it required
   with no default. MOMENTUM and QC345 still default theirs and are recorded as known offenders — a
   default that happens to agree with cockpit is not an allocation, and the day it disagrees the node
   does not boot.
3. Set `price_field="close"` for anything live. A Nautilus bar carries OHLCV and nothing else.
4. Put cadence in the **config**, never as an argument on one call path.
5. Pass `strategy_positions()` — never `positions()` — as `held`.
6. Run `dry_run()` with a seeded foreign book before wiring it anywhere.

## Every `!!` comment marks a defect that reached production

Leave them in your copy until you have deliberately considered the line each one guards. They are not
style notes; each cost a live session:

- `order_id_tag` with no default — a duplicate stops the **whole node** booting
- price field a live bar cannot supply — fails at the first rebalance, weeks later, not at boot
- warmup `max` not `sum` — QC345 reported 295 bars where 254 was right
- republished daily bar **replaces** — ~103 copies of one date gave every score `NaN`
- book state from **order events only** — a rejected entry marked held becomes a phantom
- a denied **exit** must leave the position held — otherwise nothing ever exits it
- `broker_equity` is a **method**, not a property — as a property, `None()` raises
- it reads `equity`, not `portfolio_value` — cockpit republishes Alpaca's field under that name

## What is enforced vs documented

The suite currently enforces the identity, `broker_equity` and the republished-bar rules across every
lane. The others are documented here and in the `!!` comments but **not yet asserted generically** —
found by mutation-biting this template, which is why the template earns its place: it made the gap
visible.
