# strategies

Everything for a strategy in one folder (ks#211). `strategies/<name>/` holds the pure decision
engine (`config.py`, `engine.py`, …), its Nautilus lane (`nautilus.py`; `nautilus_intraday.py` for the
intraday momentum variant), its session runner where it has one (`runner.py`), its `tests/` and its
`backtests/`. The pure layer stays deterministic, fixture-tested and importable without Nautilus;
only `nautilus*.py` and `runner.py` may import it (`strategies/tests/` asserts both).

`_layout.py` is the ONE discovery point for lanes and runners — every cross-lane conformance sweep in
`runtime/tests/` takes "every lane" from it. Shared runtime (adapter base, broker seam, mixins,
executor, store) lives under `runtime/`, which holds nothing strategy-specific.
