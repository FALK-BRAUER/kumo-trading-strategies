# momentum_rotation

Holds a small number of names currently showing strength; sells when strength fades; buys back when
it returns. Strategy id `MOMENTUM-002` (tag 002: Nautilus needs `order_id_tag` unique per trader, MANUAL holds 001).

- `config.py` — every tunable, with the measured reason for each default
- `candidates.py` — pluggable candidate sources (the pool is the edge; see issue #15)
- `score.py` — pure scale-free strength measures
- `engine.py` — gates → scores → decision. No Nautilus, no I/O.

Does NOT hold: order placement, broker calls, or anything importing `nautilus_trader` — those live
under `runtime/`.

Reading beside the code: `literature.md` (the papers behind the ranking) and `plan-intraday-two-clock.md` (the sector-into-single-name plan this lineage grew from).
