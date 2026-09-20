# ledger-book-daily-2024-2026

The momentum-rotation set over the anonymised ledger book: `bars.parquet` (daily bars, Alpaca SIP, raw/unadjusted, 2024-06 →; `manifest.json` is the exporter's record), `ledger.csv` (symbol, open, close, close%, still_open, close_source — no source identity), `ledger-snapshots.json`, `instruments.json`, `half_spreads.json`. `report.html` compares the variations; each variation is its `parameters.json` — `run.py` is identical. `bctrot` is momentum on BCTROT's clock and gap rule (lookback 40, gap 1.5%), a not-yet-successful variation, not a strategy.
