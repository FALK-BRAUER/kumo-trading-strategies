# tests/strategies

Tests for the pure decision layer — scoring, gates, candidate sources, the rotation rule.

Every test here is a bug that actually happened during research, not a hypothetical: split
contamination on raw data, leveraged ETNs dominating P&L, percentile ties breaking alphabetically,
near-zero-ATR names getting enormous share counts. Keep that standard.

Fixtures must be realistic enough to exercise what they claim. `series()` takes a `band` because
bars with `high == low == close` have zero true range, which made the ATR gate untestable and let
it sit unenforced.

Does NOT hold: anything importing nautilus — that boundary is enforced by a test.
