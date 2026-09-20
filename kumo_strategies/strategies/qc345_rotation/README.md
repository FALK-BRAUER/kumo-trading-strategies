# qc345_rotation

Pure monthly large-cap momentum rotation logic reconstructed from QC #345 research.
This package holds point-in-time feature construction, a source-based monthly universe scan, and the
final momentum ranking over whatever universe the source yields.
`QC345RotationConfig` now also exposes the shared `ExitConfig` overlay used elsewhere in the repo;
those exits are optional, default off, and are evaluated off the prior close for next-open execution.
It does not read files, call Alpaca, or bind to Nautilus; those belong in `backtesting/` and
`runtime/`.
`QC345ComputedSource.preselection_bounds()` is the audit surface for live universe derivation:
it measures how wide the input panel must be before liquidity and market-cap proxy stages select
the researched top 50. `terminal_symbols()` exposes held names known to have stopped printing so
live callers do not have to infer delisting from scanner membership.
