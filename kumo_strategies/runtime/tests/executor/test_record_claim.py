"""One writer for the ownership claim, callable from kumo-trading-platform (#540).

`exec_position_state` has rows only for BCTROT-004 and MOMENTUM-002. QC345-003 and TECHIVOL-005 hold
real positions and have never written one, so they contribute nothing to `_foreign_claims`.

THAT IS NOT COSMETIC. Measured on alpaca-paper 2026-08-25: DELL is QC345 7 + TECHIVOL 2, neither
claiming. For any lane that DOES claim DELL, `_foreign_claims` returns 0 and
`own_ceiling(acct, mine, 0)` collapses to `mine` — the attribution term VANISHES precisely when there
are two other holders. "Two claimants, no attribution" is the case `own_ceiling`'s own docstring names
as the one it exists for, and from the outside it is indistinguishable from being sole claimant. The
guard is not weakened; it is silently skipped.

ONE IMPLEMENTATION, TWO CALLERS. TECHIVOL runs on `qc27_runner` (this repo); QC345's gateway is
cockpit's. Cockpit calls this rather than reimplementing the write — two implementations of one claim
is the shape that produced the defect being fixed.
"""

from __future__ import annotations

import pytest

from kumo_strategies.runtime.executor.store import claim_upsert, drop_claim_stmt


def _set_keys(stmt) -> set[str]:
    """Columns the ON CONFLICT branch would overwrite.

    `update_values_to_set` is a list of `(column_name, value)` TUPLES, not column objects — the first
    version of this read `.name` off each entry, got the whole tuple's repr, and failed against a
    correct implementation."""
    pairs = stmt._post_values_clause.update_values_to_set
    assert pairs, "the statement has no ON CONFLICT branch — this test would prove nothing"
    return {name for name, _ in pairs}


def test_a_conflicting_write_moves_ONLY_the_quantity():
    """THE LOAD-BEARING ASSERTION. A lane with a live trail must not have it reset by a claim write.

    `entry`, `peak`, `quality` and the `sessions_*` counters are the give-back trail's memory —
    `_save_state`'s docstring: writing only `peak` "was survivable while give-back was the one rule;
    `max_hold_days` and `stall_days` count sessions and would silently reset on restart". A claim
    write that touched them would reintroduce exactly that, from a second writer, on a row the first
    writer owns.

    So the first write establishes the row and every later one moves `qty` alone.
    """
    keys = _set_keys(claim_upsert("QC345-003", "DELL", qty=7, entry_px=120.5))
    assert keys == {"qty", "updated_at"}, (
        f"the claim upsert would overwrite {sorted(keys - {'qty', 'updated_at'})} on an existing "
        f"row — that is another lane's trail memory")


def test_the_inserted_row_is_ADOPTED_and_never_invents_a_peak():
    """`quality` exists to stop exactly this. Its comment: the old seeding path wrote
    `entry, peak = (today's price, today's price)` for any position it had no row for — "asserting a
    peak that never happened". A rotation lane knows its real fill, so `entry` is true; it has no
    peak history, so `peak = entry` under `quality="adopted"`, which `TrailState.peak_is_trustworthy`
    already reads as PEAK IS NOT KNOWN and skips peak-relative rules on."""
    values = claim_upsert("TECHIVOL-005", "DELL", qty=2, entry_px=118.25).compile().params
    assert values["quality"] == "adopted"
    assert values["entry"] == 118.25
    assert values["peak"] == 118.25, "peak must equal the real entry, not a price nobody observed"
    assert values["qty"] == 2


def test_a_ZERO_claim_is_REFUSED_rather_than_written():
    """A row claiming nothing still says "this lane owns this symbol" to `_foreign_claims`, and 0 is
    exactly the value that reads as both "none" and "unset". Dropping is a different intent and has
    its own function, so the caller has to say which it means."""
    for bad in (0, 0.0):
        with pytest.raises(ValueError, match="drop_claim"):
            claim_upsert("QC345-003", "DELL", qty=bad, entry_px=120.5)


def test_a_NEGATIVE_claim_is_ACCEPTED_because_claims_are_SIGNED(): 
    """This test used to demand the opposite, and demanding it was the defect (#173).

    A short lane's claim IS negative. Refusing it forced `sync_claim_to` to write `abs(q)`, under
    which a short subtracted from every other lane's residue as though it were a competing long —
    MOMENTUM holding +30 with CRSISHORT short 10 on an account of +20 could sell only 10 of the 30
    shares it actually held. Zero is still refused, for the unchanged reason above.
    """
    stmt = claim_upsert("CRSISHORT-006", "DELL", qty=-57, entry_px=120.5)
    rendered = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "-57" in rendered, f"the sign did not reach the statement: {rendered}"


def test_a_NON_FINITE_or_absent_entry_is_REFUSED():
    """Same lesson as 43c6d3e: `nan` survives every comparison and every `or` default, and a claim
    row carrying one would poison `entry` for whatever later reads it as a real fill."""
    for bad in (float("nan"), float("inf"), 0.0, -5.0, None):
        with pytest.raises(ValueError):
            claim_upsert("QC345-003", "DELL", qty=7, entry_px=bad)


def test_the_symbol_is_NORMALISED_so_one_holding_cannot_become_two_rows():
    """`(strategy_id, symbol)` is the primary key, so `DELL` and `dell` would be two claims on one
    holding — and `_foreign_claims` SUMS by symbol, so the account would appear to owe more than it
    holds. Two callers in two repos make the casing genuinely likely."""
    a = claim_upsert("QC345-003", " dell ", qty=7, entry_px=120.5).compile().params
    assert a["symbol"] == "DELL"


def test_drop_targets_exactly_one_row():
    """A claim is dropped on a FULL exit only. A `WHERE` missing either key would clear another
    lane's claim, or every lane's claim on the symbol."""
    where = str(drop_claim_stmt("QC345-003", "DELL"))
    assert "strategy_id" in where and "symbol" in where


def test_drop_NORMALISES_the_symbol_the_same_way_the_write_does():
    """Asserted on the COMPILED PARAMS, not the SQL text — the text contains the column names
    whatever the values are, so removing `.upper()` survived that check entirely (2026-08-26).

    The consequence is not cosmetic and is worse than a failed write: `record_claim` normalises, so a
    lower-case release would target a row that does not exist, the real claim would survive, and it
    would narrow every other lane's ceiling forever. A stale claim that nothing releases is the
    #69 shape, reached through a casing mismatch across two repos."""
    params = drop_claim_stmt("QC345-003", " dell ").compile().params
    assert "DELL" in params.values(), (
        f"drop does not normalise the symbol while record does: {params} — a lower-case release "
        f"silently leaves the claim in place")
