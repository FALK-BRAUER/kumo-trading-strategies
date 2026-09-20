"""Repopulate `_held` after a restart, from the venue's own attribution (#194).

THE DEFECT. Every lane that keeps a `_held` set builds it EMPTY in `__init__` and adds to it in
exactly one place — `on_order_filled`, when the fill's intent is `enter`. Nothing else ever writes
it. So the set describes fills THIS PROCESS saw, not positions THIS LANE HOLDS, and a restart
silently replaces the second by the first.

WHAT THAT DISARMS. `contract.protective_close` opens:

    held = getattr(self, "_held", None)
    if held is None or sym not in held:
        return False

For every position predating the boot that returns False: `_record_terminal` never fires, the
venue's answer to a protective stop-out goes unrecorded, `sync_claim` is skipped, and the claim
outlives the position it describes.

MEASURED, NOT ARGUED. Two QC345-003 stop-outs at 13:36Z and 13:40Z on 2026-09-11, both after its
12:51Z recreate, both leaving claims of 3.0 against a cache of 0.0, with no journal row of any kind.
issue 133 was closed on 2026-09-10 naming exactly that observation as its criterion; it
failed nine days later. A closing criterion that names a FUTURE observation does not close a ticket,
it schedules one — which is why this module's tests assert on a restarted lane rather than on a
round trip inside one lifetime.

WHY IT IS URGENT RATHER THAN MERELY OLD. Three correct, separately-reviewed changes compose into a
hazard only when stale claims become routine:

    #178  claims stored SIGNED   ->  `other_claims` can be NEGATIVE
    #173  `reducible`            ->  the sized unwind that consumes them
    #194  `_held` never seeded   ->  stale claims after a restart are the NORMAL case

`reducible` names the reachable bad case in its own docstring — `acct=-20, mine=+10 (stale long),
other=-30 -> SELL 10` into a book that is SHORT 20 — and records that ATTRIBUTION is its only cover,
because three candidate clamps were tried and each broke a legitimate case. This module is that
attribution. It is why the cockpit pin bump is gated on it.

WHERE THE AUTHORITY COMES FROM, AND WHY NOT THE OBVIOUS ONE
-----------------------------------------------------------
CLAIMS ARE CIRCULAR AND CANNOT BE THE SOURCE. The claim is the thing that goes stale; seeding
`_held` from it would let one stale row keep re-attesting to itself forever, and the detector
(`claims_that_contradict_the_account`) would go quiet precisely when it mattered.

THE ACCOUNT'S WHOLE BOOK IS NOT OURS TO ADOPT. An unfiltered `positions_open()` returns every lane's
holdings and the operator's own. TECHIVOL-005 proposed exiting eight names belonging to two other
strategies on its first live session by reading exactly that. Adopting unattributed positions
converts this defect into the strictly worse one where a lane sells somebody else's book.

SO: the venue's attribution, filtered to us by `strategy_id`. It is already this package's idiom for
the same question (`momentum_rotation.py:862`, `qc345_rotation.py:625`, `contract.py:231`), which is
the point: the seed and the reader agree by construction rather than by coincidence.

ITS KNOWN LIMIT, STATED RATHER THAN PAPERED OVER. A position reconciled from the venue after a cold
start may carry NO strategy id, and then it is not returned here and stays invisible. That is a
smaller and safer failure than adopting it, and it is REPORTED — `_report_unattributed` logs what
the seed could see but may not claim, so an operator gets a name to look at instead of silence.
Do not "fix" that by widening the filter; the widening IS the TECHIVOL-005 defect.
"""

from __future__ import annotations

from kumo_strategies.runtime.nautilus.sides import LONG, SHORT

__all__ = ["HeldSeedMixin", "seeded_symbols", "held_symbol", "position_matches_side",
           "unattributed_positions", "position_is_unattributed"]


def held_symbol(position) -> str | None:
    """The symbol string for `position`, spelled the way `protective_close` reads it.

    `contract.py` does `str(event.instrument_id.symbol)` and the lanes do
    `pos.instrument_id.symbol.value`. Those agree on a Nautilus `Symbol`, but the seed and the
    reader agreeing BY ACCIDENT is how #194 happened; `str()` is taken over the same attribute so a
    disagreement would have to be introduced deliberately.
    """
    try:
        return str(position.instrument_id.symbol)
    except Exception:                                              # noqa: BLE001
        return None


def position_is_unattributed(position) -> bool:
    """Does `position` carry NO strategy attribution?

    NAUTILUS SPELLS THIS AS A VALUE, NOT AN ABSENCE. `Position.strategy_id` is ALWAYS a
    `StrategyId`; a position that "did not originate from any strategy being managed by the system"
    carries `StrategyId("EXTERNAL")` and answers `is_external()` True. The first version of this
    tested `strategy_id in (None, "")` and was wrong twice over, both found by the paper tenant
    owner driving it in a container rather than reading it:

      IT RAISED ON EVERY ATTRIBUTED POSITION. `StrategyId.__eq__` is Cython-typed, so `sid == ""`
      raises TypeError (`expected StrategyId, got str`) — verified here, not quoted. The `None` arm
      is fine; the tuple reaches the `""` arm on every position carrying any attribution at all.

      AND THE SENTINEL WAS WRONG. Nothing ever carries None, so fixing only the raise would have
      left the probe returning ZERO on both tenants forever — and the bump gate that reads it would
      have been satisfied VACUOUSLY by a probe that cannot see its subject. That is the worse of the
      two failures: a raise gets investigated, a confident zero gets accepted.

    MY OWN TESTS COULD NOT HAVE CAUGHT EITHER, and that is the lesson worth keeping. The doubles
    held plain strings, and a double that cannot REPRESENT a `StrategyId` cannot express the bug.
    The fixtures now carry real ones.

    `is_external()` is asked for BY IDENTITY OF BEHAVIOUR, not by importing a constant:
    `EXTERNAL_STRATEGY_ID` exists in `identifiers.pyx` but is NOT exported from the compiled module
    (verified — `ImportError` on this pin), so importing it would break on the machine it is meant
    to run on. The string fallback keeps plain-string doubles and non-Nautilus callers working.
    """
    sid = getattr(position, "strategy_id", None)
    if sid is None:
        return True
    probe = getattr(sid, "is_external", None)
    if callable(probe):
        try:
            return bool(probe())
        except Exception:                                              # noqa: BLE001
            pass
    return str(sid).strip() in ("", "EXTERNAL")


def position_matches_side(position, side) -> bool:
    """Does `position` sit on the side this lane holds?

    A SHORT lane that adopted a LONG position would hand `protective_close` a name whose closing
    side is inverted, and it would then ignore every real protective close on it — the #194 defect,
    inverted, which is the failure mode `contract.py` already refuses to default its way into.

    A zero position is FLAT and matches nothing: seeding it would make the lane assert a holding it
    does not have, which is the mirror of the blindness being fixed. Both comparisons below exclude
    it, so there is no separate zero guard — one was written here and a mutation bite proved it dead
    code, which is the same unearned-line shape this repo files against tests.

    AN UNRECOGNISED SIDE IS REFUSED, not assumed. Written as `qty > 0 if side == LONG else qty < 0`
    this would read every non-LONG value — including None, a string, a typo — as SHORT, and quietly
    adopt short positions into a lane whose side nobody established. `contract.protective_close`
    refuses rather than defaults at exactly this point, for exactly this reason.
    """
    try:
        qty = float(position.signed_qty)
    except Exception:                                              # noqa: BLE001
        return False
    # FLOAT, NOT INT. `int(0.5) == 0`, so truncating would read a fractional holding as FLAT and
    # leave it out of `_held` — the #194 symptom surviving the #194 fix, for exactly the positions
    # nobody would think to check. Not live today (zero fractional positions on either tenant) but
    # Alpaca supports fractional shares, so this is latent rather than impossible.
    if side == LONG:
        return qty > 0
    if side == SHORT:
        return qty < 0
    return False


def seeded_symbols(positions, side) -> set[str]:
    """The symbols a lane holding `side` may adopt from `positions`. Pure, so it is testable alone."""
    out: set[str] = set()
    for p in positions or ():
        if not position_matches_side(p, side):
            continue
        sym = held_symbol(p)
        if sym:
            out.add(sym)
    return out


def unattributed_positions(all_positions, mine=()) -> list[str]:
    """Symbols among `all_positions` that carry NO strategy attribution. Pure, and that is the point.

    THIS IS READABLE BEFORE THE PIN THAT CONTAINS IT. `_report_unattributed` runs inside `on_start`,
    so its count only exists once a lane has booted on a revision carrying this module — which is
    AFTER a bump, not before it. A gate that must be read at the bump cannot be satisfied by a
    number that only appears afterwards, so the measurement is a free function over a position list:
    kumo-trading-platform can run it out-of-band against the live book on the CURRENT pin, the same way the
    `claims_that_contradict_the_account` probe is run.

    NOT A BROKER CALL, and cannot become one. It takes positions as an argument and reads no
    connector, no REST client and no venue. Whoever holds the Nautilus cache supplies the list.

    AN EMPTY INPUT IS NOT AN ANSWER OF ZERO. A caller that hands this an empty list because its
    enumeration failed would read "no unattributed positions" and be reassured by a broken probe —
    the false-14-row incident in the other direction. `None` raises; an empty list returns empty and
    the caller is responsible for a positive control, which is why this returns a LIST OF NAMES
    rather than a count: a name can be checked against the book, a zero cannot.
    """
    if all_positions is None:
        raise ValueError(
            "unattributed_positions(None) — an enumeration that failed must not read as zero "
            "unattributed positions. Pass the list, or let the failure raise where it happened.")
    ours = {id(p) for p in mine or ()}
    return sorted({s for p in all_positions
                   if id(p) not in ours and position_is_unattributed(p)
                   and (s := held_symbol(p))})


class HeldSeedMixin:
    """Gives a lane the ability to recognise positions it did not personally fill.

    A MIXIN RATHER THAN SEVEN COPIES, because the defect is that seven lanes independently got the
    same thing wrong, and seven copies of the fix would be seven chances to drift apart. The same
    reasoning as `MarketAwareMixin`.
    """

    def seed_held_from_positions(self) -> set[str]:
        """Adopt this lane's own open positions into `_held`. Returns what was newly adopted.

        NEVER RAISES, and that is load-bearing rather than defensive habit: this runs in `on_start`,
        and a lane that dies there never arms, never decides and reports nothing — strictly worse
        than the blindness it is fixing. The same rule `_record_arm_state` learned the hard way when
        its `clock.timestamp_ns()` raised over the arming failure it was reporting.

        ADDITIVE, never a replacement. `_held` may already hold names from fills seen this process
        (a restart mid-session, a reconnect), and those are not re-derivable from open positions —
        a name entered and exited this session is correctly held by neither. `|=`, not `=`.
        """
        side = getattr(self, "POSITION_SIDE", None)
        if side is None:
            # Same refusal as `protective_close`: a lane must state which side it holds before its
            # positions can be interpreted. Reported, not raised, because of the rule above.
            self._seed_log("error", f"{type(self).__name__} has no POSITION_SIDE; "
                                    f"`_held` cannot be seeded and protective closes stay blind")
            return set()
        held = getattr(self, "_held", None)
        if held is None:
            # A lane with no `_held` keeps no lane book by design (penny_gap declares `NO_LANE_BOOK`
            # and its `protective_close` is deliberately inert). Nothing to seed, and nothing wrong.
            return set()
        try:
            mine = list(self.cache.positions_open(strategy_id=self.id))
        except Exception as exc:                                   # noqa: BLE001
            self._seed_log("error", f"could not read open positions to seed `_held` ({exc!r}); "
                                    f"protective closes on pre-boot positions will go unrecorded")
            return set()

        adopted = seeded_symbols(mine, side) - set(held)
        held |= adopted
        if adopted:
            self._seed_log("info", f"seeded `_held` with {len(adopted)} position(s) held before this "
                                   f"start: {sorted(adopted)}")
        self._report_unattributed(mine)
        return adopted

    def _report_unattributed(self, mine) -> None:
        """Name the open positions this lane may NOT adopt, so the limit is visible, not silent.

        Reconciliation after a cold start can leave a position with no strategy id. Such a position
        is invisible to the seed above and its protective close will still go unrecorded. That is
        deliberate — adopting it is the TECHIVOL-005 defect — but an operator must get a NAME rather
        than silence, because "nothing in the journal" is exactly what #133 looked like.

        ONLY THE UNATTRIBUTED ONES. A position carrying ANOTHER lane's id is correctly not ours and
        is that lane's business; naming it here would print most of the account's book into this
        lane's log every single start, and a warning that fires every start is one nobody reads —
        which would cost exactly the signal this method exists to provide.
        """
        try:
            everything = list(self.cache.positions_open())
        except Exception:                                          # noqa: BLE001
            return                                                 # best-effort reporting only
        orphans = unattributed_positions(everything, mine)
        if orphans:
            self._seed_log("warning",
                           f"{len(orphans)} open position(s) carry no strategy attribution and were "
                           f"NOT adopted: {orphans}. A protective close on these goes unrecorded "
                           f"(#194); adopting them would let this lane act on another lane's book.")

    def _seed_log(self, level: str, msg: str) -> None:
        """Log without ever becoming the reason `on_start` failed."""
        try:
            getattr(self.log, level)(msg)
        except Exception:                                          # noqa: BLE001
            pass
