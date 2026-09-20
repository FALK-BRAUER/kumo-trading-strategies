# tests/fixtures

Small test inputs that happen to be data-shaped: a few daily closes for a handful of names, cut
to the sessions a test needs. `bin/check-public-tree.sh` exempts `tests/fixtures/` from its
"no data file tracked" rule, and `tests/public/test_no_tracked_data_files.py` caps a fixture at
64 KiB — larger is a dataset and belongs in research-data. Goes here: one subdirectory per
strategy. Does not go here: panels, results, anything a research script produced.
