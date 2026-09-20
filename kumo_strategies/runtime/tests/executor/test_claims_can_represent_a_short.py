"""The claims ledger has to answer "how much may I unwind" for a SHORT lane too (#173).

`own_ceiling(acct_qty, my_claim, other_claims) = max(0, min(mine, acct - other))` asks HOW MUCH MAY
I SELL, and `acct - other` is the right arithmetic only while every lane is long and selling is the
only way to reduce. CRSISHORT holds the other side, so its reducing action is a BUY, and the
function is being asked a question it cannot express.

BOTH SIGN CONVENTIONS FAIL, so this is not a sign flip. Measured on the function itself:

    MOMENTUM +30 AAPL, CRSISHORT short 10, account +20
      claim stored as MAGNITUDE (what `store.py` writes today)  -> MOMENTUM's ceiling 30 -> 10
      claim stored SIGNED                                       -> MOMENTUM's ceiling 30  (right)
    CRSISHORT covering its own short, account -50
      either convention                                         -> 0, because the floor is
                                                                   `max(0, ...)` and `mine` is
                                                                   negative

Magnitude freezes a long lane's exit on any shared symbol — silently, because the symbol then fails
`if q > 0` in `pgrunner.run` and drops out of `held_qty`, so give-back, stall, forced exits and
LIQUIDATING all skip it. That is the kumo-trading-platform issue 197 B8 shape: a permanent short nothing looked for.

WHAT MUST SURVIVE, and it is the half a "just make it signed" fix destroys: `acct - other` is the
only bound a lane has against ITS OWN STALE CLAIM. Claims record on ACCEPT, before the venue
answers — MOMENTUM-002 once carried a 260-share LAND claim on a position that never existed, and
TECHIVOL-005 claimed 97 TOST against a book of 0. Drop the term and a lane with a phantom +28 sells
28 shares belonging to the lane that actually holds them (#88, the WHD incident).

THE ANSWER CARRIES ITS SIDE, and that is not decoration. `pgrunner` is LONG-ONLY: it feeds the
result into `held_qty` and sells it. A function that returned a short lane's cover size as a bare
positive integer would make the long-only runner SELL a short position — doubling it. So the
reducing direction travels with the quantity and a long-only caller cannot receive a cover.
"""

from __future__ import annotations

import pytest

from kumo_strategies.strategies.momentum_rotation.runner import own_ceiling


def _reducible():
    fn = None
    try:
        from kumo_strategies.strategies.momentum_rotation.runner import reducible as fn
    except ImportError:
        pass
    if fn is None:
        pytest.fail(
            "`reducible` does not exist: the claims ledger can express how much a LONG lane may "
            "sell and has no way to express how much a SHORT lane may buy to cover (#173)")
    return fn


# -- the two failures, stated as the answers they should give ---------------------------------------

def test_a_SHORT_lane_can_size_its_own_cover():
    """The lone short. `own_ceiling(-50, -50, 0)` is 0 today, so a short lane cannot cover at all."""
    r = _reducible()(-50, -50, 0)
    assert r.qty == 50, f"a lane short 50 against an account of -50 may cover 50, got {r.qty}"
    assert r.side == "BUY", "covering a short is a BUY; a SELL would double the position"


def test_a_SHORT_claim_does_not_FREEZE_a_long_lane_on_the_same_symbol():
    """MOMENTUM +30, CRSISHORT short 10, account +20. MOMENTUM owns 30 and may sell 30.

    Under magnitude storage its ceiling is 10 — it cannot sell 20 shares it holds — and the symbol
    then drops out of `held_qty` entirely.
    """
    r = _reducible()(20, 30, -10)
    assert (r.qty, r.side) == (30, "SELL"), r


def test_each_lane_reduces_INDEPENDENTLY_on_one_instrument():
    """Both directions on one symbol, which is the case both conventions got wrong. Neither lane's
    ability to unwind is reduced by the other's claim, and the aggregate still reconciles."""
    red = _reducible()
    momentum = red(20, 30, -10)
    crsishort = red(20, -10, 30)
    assert (momentum.qty, momentum.side) == (30, "SELL")
    assert (crsishort.qty, crsishort.side) == (10, "BUY")
    assert 30 + (-10) == 20, "the fixture itself must reconcile with the account"


# -- what must NOT be lost --------------------------------------------------------------------------

def test_a_STALE_claim_still_cannot_reach_into_another_lanes_position():
    """THE #88 BREACH, and the reason `acct - other` survives.

    `broker.py` documents this exact case: mine 28, other 28, account 28. Our claim is a phantom —
    the position is entirely the other lane's. Anything but 0 sells their shares.
    """
    r = _reducible()(28, 28, 28)
    assert r.qty == 0, (
        f"a lane with a phantom claim was allowed to reduce {r.qty} — those shares belong to the "
        f"lane that actually holds them (#88, the WHD incident)")


def test_the_documented_FREEZE_numbers_are_unchanged():
    """The two worked examples in `retire_claims`' docstring, which describe a real incident. A
    change to this arithmetic that moved them would be rewriting history, not fixing a bug."""
    red = _reducible()
    assert red(79, 79, 77).qty == 2
    assert red(79, 77, 79).qty == 0


@pytest.mark.parametrize("acct,mine,other,want", [
    (100, 100, 0, 100),      # the ordinary long case
    (-93, 100, 0, 0),        # a LONG claim against a short book: still nothing to sell
    (0, 28, 28, 0),          # signed account, both claims: correct today and must stay
    (56, 28, 28, 28),        # the unsigned-account case broker.py warns about
])
def test_every_long_answer_is_BYTE_IDENTICAL_to_today(acct, mine, other, want):
    """The generalisation must not move a single long-lane number. Asserted against `own_ceiling`
    as it stands so the two are compared rather than one being trusted."""
    assert own_ceiling(acct, mine, other) == want
    r = _reducible()(acct, mine, other)
    assert r.qty == want and (r.side == "SELL" or r.qty == 0)


# -- the long-only runner must never be handed a cover ---------------------------------------------

def test_own_ceiling_STILL_answers_zero_for_a_short_claim():
    """`pgrunner` is long-only and feeds this straight into `held_qty`, which it SELLS.

    A short lane's cover size arriving here as a bare positive integer would make the long-only
    runner sell a short position — doubling it. `own_ceiling` keeps its long-only meaning and
    delegates the arithmetic, so the long-only caller physically cannot receive a cover.
    """
    assert own_ceiling(-50, -50, 0) == 0
    assert own_ceiling(-20, -50, 30) == 0
    assert _reducible()(-50, -50, 0).qty == 50, "the capability exists, just not through this door"


# -- the aggregate invariant is a DETECTOR, never a handcuff ----------------------------------------

def test_a_SHORT_over_claim_is_REPORTED_where_the_original_test_missed_it():
    """`over_claimed` used `total > account`, which is the breach only while every claim is
    positive. A lane claiming -50 against an account of -20 has over-claimed by 30 shares in exactly
    the same way, and `-50 > -20` is False — so the worst short breach read as a consistent ledger.

    Extended rather than joined by a sibling detector: two functions answering "does the ledger
    agree with the venue" is two records of one fact, and they disagree silently.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import over_claimed

    assert over_claimed({"AAA": 20}, {"MOMENTUM": {"AAA": 30}, "CRSISHORT": {"AAA": -10}}) == {}, (
        "a reconciling ledger (+30 -10 == +20) was reported as broken")

    short_breach = over_claimed({"AAA": -20}, {"CRSISHORT": {"AAA": -50}})
    assert "AAA" in short_breach, (
        "a lane claiming -50 against an account of -20 was not reported — this is the case the "
        "unsigned `>` test could never see")
    assert short_breach["AAA"] == (-50, -20)

    long_breach = over_claimed({"AAA": 20}, {"MOMENTUM": {"AAA": 30}})
    assert long_breach["AAA"] == (30, 20), "the original long breach must still fire"


def test_UNDER_claiming_is_NOT_a_breach():
    """A position reconciled in after a restart carries no lane, so the claims sum to LESS than the
    account. That is routine, not a breach — and `reducible`'s residue term already keeps it safe:
    shares nobody claims are shares no lane may reach for."""
    from kumo_strategies.strategies.momentum_rotation.runner import over_claimed

    assert over_claimed({"AAA": 50}, {"MOMENTUM": {"AAA": 20}}) == {}, (
        "unattributed shares were reported as an over-claim")


def test_a_BROKEN_ledger_does_NOT_stop_a_lane_getting_out():
    """A lane must ALWAYS be able to unwind its own position. If the ledger is wrong that is an
    alarm, not a handcuff — the freeze this ticket exists to remove was exactly a disagreement
    silently becoming a refusal."""
    from kumo_strategies.strategies.momentum_rotation.runner import over_claimed

    assert over_claimed({"AAA": -20}, {"MOMENTUM": {"AAA": 30}}), "the fixture must be a breach"
    # ...and the short lane can still cover every share the venue actually shows.
    assert _reducible()(-20, -50, 30).qty == 50


# -- what SIGNING made reachable, and what covers it (ffv73l93's review of 0c51d34) ------------------

def test_a_STALE_LONG_claim_can_SELL_INTO_A_SHORT_BOOK_and_the_docstring_must_say_so():
    """THE ASYMMETRY THE FIRST DOCSTRING DENIED.

    It claimed `-residue` is positive only when the residue is genuinely SHORT, "the mirror of a
    long lane being unable to sell against a short one". That is false: subtracting a neighbour's
    SIGNED short INCREASES a long lane's residue.

        acct=-20  mine=+10 (stale long)  other=-30  ->  residue +10  ->  SELL 10

    The book is SHORT 20 and this sells 10 more of it. Under magnitude storage `other` was +30, the
    residue was -50, and the answer was 0 — SO THE SIGN CHANGE MADE IT REACHABLE. It is my defect,
    introduced by the fix.

    IT IS NOT CLAMPABLE. Three clamps were tried upstream and each broke
    `MOM +30, CRSI -10, acct +20 -> SELL 30`, because the arithmetic cannot tell a phantom `mine`
    from a real one: both are a positive claim against an account whose sign differs.

    WHAT COVERS IT IS ATTRIBUTION. With `mine` populated, `pgrunner` narrows by
    `q = min(q, attributed_qty)`; the attributed quantity for a lane holding none of a short book is
    not positive, so the symbol drops. The exposure is exactly the no-attribution case — a position
    reconciled in after a restart — which this package explicitly supports, plus a stale claim,
    which its own docstrings call routine.

    So this test pins the behaviour as KNOWN rather than fixed, and `claims_that_contradict_the_account`
    is what surfaces it when attribution is absent.
    """
    r = _reducible()(-20, 10, -30)
    assert (r.qty, r.side) == (10, "SELL"), (
        "the stale-long-into-a-short-book case changed. If it is now 0, delete this test and the "
        "asymmetry paragraph in `reducible`'s docstring — but check the MOM/CRSI case first.")

    mirror = _reducible()(20, -10, 30)
    assert (mirror.qty, mirror.side) == (10, "BUY"), "the mirror moved"

    import inspect

    from kumo_strategies.strategies.momentum_rotation.runner import reducible
    doc = inspect.getdoc(reducible) or ""
    assert "ATTRIBUTION" in doc.upper(), (
        "`reducible`'s docstring does not name attribution as what covers the stale-claim case — "
        "and its first version claimed a symmetry that does not hold")


def test_a_NET_ZERO_claim_set_is_NOT_a_breach():
    """ZERO IS NOT A SIDE — the rule `reducible`'s own third branch already applies.

    `total > net if total >= 0 else total < net` put `total == 0` on the LONG branch, so a claim set
    that nets to zero against a short account reported a breach while claiming NOTHING against the
    same account reported clean. Same real state, two answers — and under-claiming is documented
    here as not-a-breach.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import over_claimed

    netted = over_claimed({"X": -5}, {"A": {"X": 10}, "B": {"X": -10}})
    nothing = over_claimed({"X": -5}, {})
    assert netted == nothing == {}, f"net-zero {netted} against nothing-claimed {nothing}"


def test_a_claim_whose_SIGN_CONTRADICTS_THE_ACCOUNT_is_reported():
    """TOTALS HIDE THE COMPOSITION. `over_claimed` sums, so the exact state behind the stale-long
    case reports clean: `{A:+10 phantom, B:-30}` against an account of -20 totals to -20 == net.

    A per-lane check sees it in one comparison — a lane claiming LONG against a SHORT account holds
    nothing of it. This is a DETECTOR and never a gate, so a false positive costs a row rather than
    a frozen exit, and it is the only thing that surfaces the stale-long case when attribution is
    absent.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import claims_that_contradict_the_account

    got = claims_that_contradict_the_account(
        {"X": -20}, {"A": {"X": 10}, "B": {"X": -30}})
    assert got == {("A", "X"): (10, -20)}, got

    # the same composition that `over_claimed` calls clean
    from kumo_strategies.strategies.momentum_rotation.runner import over_claimed
    assert over_claimed({"X": -20}, {"A": {"X": 10}, "B": {"X": -30}}) == {}


def test_the_contradiction_detector_does_NOT_fire_on_an_ordinary_book():
    """THE CONTROL. Every lane long against a long account, and a short lane against a short
    account, must be silent — or the detector is noise and gets switched off."""
    from kumo_strategies.strategies.momentum_rotation.runner import claims_that_contradict_the_account

    assert claims_that_contradict_the_account(
        {"X": 30}, {"MOMENTUM": {"X": 20}, "BCTROT": {"X": 10}}) == {}
    assert claims_that_contradict_the_account(
        {"X": -50}, {"CRSISHORT": {"X": -50}}) == {}
    # a FLAT account cannot contradict anything: zero is not a side
    assert claims_that_contradict_the_account({"X": 0}, {"A": {"X": 10}}) == {}
