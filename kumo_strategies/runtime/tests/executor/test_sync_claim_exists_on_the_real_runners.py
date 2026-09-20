"""`sync_claim` has to exist on the RUNNERS THIS REPO SHIPS, not only on cockpit's gateways (#133).

`RegistrationMixin._sync_claim_to_book` reaches the runner through `getattr(runner, "sync_claim",
None)` and returns quietly when it is absent. That is the right shape for a live event handler — a
missing method must not raise into Nautilus's dispatch — and it means an ENTIRELY ABSENT method is
indistinguishable from a working one at the call site.

It was absent. `grep -rn "def sync_claim" src/` returned nothing across this whole package: only
kumo-trading-platform's gateways defined it, and `QC27SessionRunner` is the runner TECHIVOL-005 actually runs
on — the lane the #133 incident happened to. So the claim re-sync was inert on the one lane that
needed it, and the #133 test suite passed anyway, because the `_Runner` double in that file defines
`sync_claim`.

THAT IS A DOUBLE MORE CAPABLE THAN PRODUCTION, and it manufactured a passing test for a code path
that could not run. The lesson this package keeps paying for is the same one: a fake that is more
forgiving than the real thing hides the defect it was written to catch.

So these tests bind the REAL runner classes. `hasattr` on the shipped class is the fixture-property
that the #133 suite could not express, and the behaviour below is driven through a ledger double
that records the statements rather than a runner double that pretends.
"""

from __future__ import annotations

import asyncio
import re
import inspect

import pytest

from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
from kumo_strategies.strategies.qc27_tech_inverse_vol.runner import QC27SessionRunner
from kumo_strategies.runtime.executor.store import sync_claim_to

#: Every runner an adapter in this package can be handed. A runner added later without `sync_claim`
#: fails here rather than being discovered inert during an incident.
RUNNERS = [PgSessionRunner, QC27SessionRunner]


@pytest.mark.parametrize("cls", RUNNERS, ids=lambda c: c.__name__)
def test_the_shipped_runner_defines_sync_claim(cls):
    """THE SEAM. `getattr(runner, "sync_claim", None)` cannot tell absent from working, so absence
    has to fail HERE — nothing downstream can see it."""
    assert hasattr(cls, "sync_claim"), (
        f"{cls.__name__} has no sync_claim, so every claim re-sync through it is silently inert")
    assert inspect.iscoroutinefunction(cls.sync_claim), (
        f"{cls.__name__}.sync_claim must be a coroutine — the adapters hand it to `fire_and_report`")


@pytest.mark.parametrize("cls", RUNNERS, ids=lambda c: c.__name__)
def test_its_signature_matches_what_the_adapter_calls(cls):
    """The adapter calls `sync_claim(symbol, qty, px)` positionally. A runner that named its
    parameters differently would still bind, but one that took fewer would raise inside a live event
    handler — which is where this must never happen."""
    params = [p for p in inspect.signature(cls.sync_claim).parameters if p != "self"]
    assert params[:3] == ["symbol", "qty", "px"], params


# -- the behaviour, through a ledger double that records statements -----------------------------

class _Ledger:
    """Records what was written, and refuses what the real one refuses."""

    def __init__(self):
        self.writes = []

    async def write_claim(self, journal, strategy_id, symbol, stmt):
        self.writes.append((strategy_id, symbol, stmt))


async def _drive(qty, px, *, ledger, strategy_id="TECHIVOL-005", symbol="TOST"):
    return await sync_claim_to(None, strategy_id, symbol, qty, px, write=ledger.write_claim)


def test_a_flat_position_DROPS_the_claim():
    """The #133 incident: 97 claimed against a book of 0 for a whole session."""
    led = _Ledger()
    asyncio.run(_drive(0, 36.5, ledger=led))
    assert len(led.writes) == 1
    _sid, sym, stmt = led.writes[0]
    assert sym == "TOST"
    assert "DELETE" in str(stmt).upper(), f"a flat position did not drop the claim: {stmt}"


def test_a_PARTIAL_moves_the_quantity_and_does_not_drop_it():
    """A partial exit that dropped the claim would make the lane stop claiming shares it still
    holds — which loosens every OTHER lane's ceiling onto our position, the mirror of the #133
    breach rather than a fix for it."""
    led = _Ledger()
    asyncio.run(_drive(57, 36.5, ledger=led))
    assert len(led.writes) == 1
    _sid, _sym, stmt = led.writes[0]
    text = str(stmt).upper()
    assert "DELETE" not in text, "a partial exit dropped the claim"
    assert "INSERT" in text or "UPDATE" in text, text
    # THE QUANTITY MUST REACH THE STATEMENT. "not DELETE" was satisfied by a write that moved the
    # claim to the WRONG number — that is exactly how `abs(q)` survived here for as long as it did
    # (#173): the claim was written, at the wrong value, and this assertion could not see it.
    assert "57" in str(stmt.compile(compile_kwargs={"literal_binds": True})), (
        "the partial's new quantity did not reach the statement")


def test_REPEATED_partials_need_NO_delta_API_because_sync_is_ABSOLUTE():
    """issue 177's only trade is a partial resize ~93 times a year, which raised whether
    the ledger needs an ADJUST-BY-DELTA call. It does not, and adding one would be a defect.

    `sync_claim_to` takes the BOOK's quantity and sets the claim to it. That is idempotent and
    self-correcting: every call re-states the truth, so a missed call, a duplicate call or a
    reordered pair all converge. A delta API accumulates instead — one dropped `-10` leaves the
    claim permanently wrong with nothing to compare against, and it would be a SECOND way to move
    one number, which is how two conventions arrive and no row says which it is.

    Driven as the sequence the sleeve actually produces: 100 -> 93 -> 88 -> 88 (no trade) -> 0.
    """
    led = _Ledger()
    for qty in (100, 93, 88, 88, 0):
        asyncio.run(_drive(qty, 36.5, ledger=led))

    rendered = [str(stmt.compile(compile_kwargs={"literal_binds": True}))
                for _sid, _sym, stmt in led.writes]
    assert len(rendered) == 5, f"a resize did not reach the ledger: {len(rendered)} writes"
    for want, text in zip(("100", "93", "88", "88"), rendered[:4]):
        assert want in text, f"claim did not move to {want}: {text}"
    assert "DELETE" in rendered[-1].upper(), "the flat close did not drop the claim"

    # IDEMPOTENT IN THE QUANTITY, which is the only field that carries state. The statements
    # differ in `updated_at` and must — comparing them whole asserted that the clock does not
    # advance, which is a property of the test, not of the ledger.
    #
    # The third and fourth syncs re-state the SAME number: absolute, so nothing moves. A delta API
    # handed the same pair would have applied the resize twice and taken 88 to 0.
    qtys = [re.search(r"DO UPDATE SET qty = ([\-0-9.]+)", t).group(1) for t in rendered[:4]]
    assert qtys == ["100.0", "93.0", "88.0", "88.0"], qtys


def test_a_SHORT_position_is_claimed_SIGNED_and_never_dropped():
    """CRSISHORT holds negative quantities. `qty <= 0` would read a live 57-share short as flat and
    release the claim on a real position — the exact defect this method exists to prevent, mirrored.
    Only ZERO is flat.

    SIGNED, NOT MAGNITUDE (#173), and the rename is the point rather than tidiness. Under `abs(q)` a
    short lane's claim subtracted from every other lane's residue as though it were a competing
    LONG: MOMENTUM holding +30 with CRSISHORT short 10 on an account of +20 could sell only 10 of
    the 30 shares it actually held, and the symbol then failed `if q > 0` in `pgrunner.run` and
    vanished from `held_qty`, so give-back, stall, forced exits and LIQUIDATING all skipped it.
    """
    led = _Ledger()
    asyncio.run(_drive(-57, 36.5, ledger=led))
    assert len(led.writes) == 1
    _sid, _sym, stmt = led.writes[0]
    assert "DELETE" not in str(stmt).upper(), "released the claim on a live SHORT position"
    # THE SIGN REACHES THE STATEMENT. Asserting only "not dropped" was satisfied by `abs(q)` for as
    # long as that was the bug — the claim was written, at the wrong sign, and this test was green.
    assert "-57" in str(stmt.compile(compile_kwargs={"literal_binds": True})), (
        "the claim was written UNSIGNED: a short stored as +57 subtracts from every other lane's "
        "residue as though this lane were long")


def test_a_MISSING_price_does_not_fabricate_an_entry():
    """`claim_upsert` moves `qty` alone on conflict and uses `entry_px` only on INSERT. With no price
    and no existing row, writing anyway would invent an entry price that every later reader treats
    as observed — a fabricated value is worse than an absent one, because nothing downstream can
    tell it apart. The row is left alone and the caller is told."""
    led = _Ledger()
    wrote = asyncio.run(_drive(57, None, ledger=led))
    assert led.writes == [], "fabricated an entry price for a claim it could not price"
    assert wrote is False


def test_a_flat_position_still_drops_WITHOUT_a_price():
    """Dropping needs no price. Refusing to drop for a missing price would leave the #133 breach
    standing in exactly the case that matters most — a protective fill carries a price, but a
    reconciled or synthesised terminal event may not."""
    led = _Ledger()
    wrote = asyncio.run(_drive(0, None, ledger=led))
    assert len(led.writes) == 1 and wrote is True
    assert "DELETE" in str(led.writes[0][2]).upper()
