"""The declared envelopes are bound to the configs they were fitted on (#147).

`envelope.py` enforces the four RULES. This file enforces that the two lanes actually declared here
obey them with their real numbers — which is a different claim, and the one that breaks first when
somebody edits a config.

THE FAILURE THIS PREVENTS: a lane's schedule, sizing or exit rule changes, its behaviour changes with
it, and the old distribution stays. Every percentile it then reports describes a strategy that is no
longer running, while carrying the authority of a number. `rebalance_period` is the precedent — a
sweep reported weekly while both production call sites stayed hardwired monthly, and nothing failed
to say so.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from kumo_strategies.strategies import envelope as envelope_mod
from kumo_strategies.strategies.envelope import (
    IN_ENVELOPE, OUT_OF_ENVELOPE, StaleEnvelope, assess)
from kumo_strategies.strategies.momentum_rotation import envelopes as reg
from kumo_strategies.strategies.momentum_rotation.config import (
    ExecutionConfig, ExitConfig, MomentumRotationConfig, PortfolioConfig)


def _cfg(slots, **kw):
    return MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=kw.pop("n_hold", 8), buffer=kw.pop("buffer", 5)),
        exits=ExitConfig(give_back_frac=kw.pop("give_back_frac", 0.5)),
        execution=ExecutionConfig(decision_slots=slots),
    )


MOM = ("open+5m",)
BCT = ("open+5m", "open+150m", "close-20m")


# -- the fingerprint is DERIVED, which is what makes the VERSIONED rule real ------------------------

def test_the_fingerprint_is_COMPUTED_from_the_config_not_written_down():
    """A version string typed next to an envelope agrees with the config because a human said so.
    That is this package's recurring defect: a knob matching its configured value because nothing
    recomputed it. Asserted on the AST — `fingerprint` must actually hash something derived from
    `cfg`, not return a literal."""
    tree = ast.parse(inspect.getsource(reg.fingerprint))
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    assert returns, "fingerprint returns nothing"
    assert not any(isinstance(r.value, ast.Constant) for r in returns), (
        "fingerprint returns a literal — the VERSIONED rule is then decorative")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "cfg" in names, "fingerprint does not read the config it claims to describe"


@pytest.mark.parametrize("change", [
    {"slots": ("open+150m",)},
    {"n_hold": 6},
    {"buffer": 3},
    {"give_back_frac": 0.15},
])
def test_ANY_behaviour_bearing_change_moves_the_fingerprint(change):
    """Each of these changes what the lane DOES, so each must invalidate the distribution. A field
    that does not move the fingerprint is a config change the envelope cannot notice."""
    slots = change.pop("slots", MOM)
    before = reg.fingerprint(_cfg(MOM))
    after = reg.fingerprint(_cfg(slots, **change))
    assert before != after, f"{change or slots} left the fingerprint unchanged"


def test_the_GAP_RULE_IS_READ_OFF_THE_CONFIG_NOT_PASSED_IN():
    """A lane running `min_abs_gap_pct` has different behaviour — 45% of candidates removed — and
    must not read the gap-off bands as describing it.

    IT MUST COME FROM THE CONFIG. It was a separate `gap_rule=` argument until a mutation showed
    that argument was dead: removing the gap from the fitted config changed no fingerprint, because
    the flag was supplied by hand beside it. Two derivations of one fact, free to disagree — the
    config saying "no gap" while the fingerprint said "gap".
    """
    import inspect
    assert "gap_rule" not in inspect.signature(reg.fingerprint).parameters, (
        "the gap rule is a caller argument again — it can disagree with the config it describes")
    off = _cfg(BCT)
    on = MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=8, buffer=5),
        exits=ExitConfig(give_back_frac=0.5),
        execution=ExecutionConfig(decision_slots=BCT, min_abs_gap_pct=0.015))
    assert reg.fingerprint(off) != reg.fingerprint(on)


def test_the_gap_THRESHOLD_ITSELF_MOVES_THE_FINGERPRINT():
    """Not merely on/off. 1.25%, 1.5% and 1.75% produce measurably different lanes — turnover 1.42x,
    1.35x, 1.29x against MOMENTUM — so an envelope fitted at one threshold does not describe another."""
    def at(g):
        return reg.fingerprint(MomentumRotationConfig(
            portfolio=PortfolioConfig(n_hold=8, buffer=5),
            exits=ExitConfig(give_back_frac=0.5),
            execution=ExecutionConfig(decision_slots=BCT, min_abs_gap_pct=g)))
    assert len({at(0.0125), at(0.015), at(0.0175)}) == 3, (
        "different gap thresholds share a fingerprint — a lane could read another threshold's bands")


def test_the_two_lanes_have_DIFFERENT_fingerprints():
    """Same pool, same exits, same engine — only the clock differs, and the clock is the whole
    reason they have separate distributions. Identical fingerprints would let each read the
    other's bands."""
    assert reg.fingerprint(_cfg(MOM)) != reg.fingerprint(_cfg(BCT))


# -- the declared envelopes are bound to their own configs -------------------------------------------

@pytest.mark.parametrize("lane,slots", [("MOMENTUM-002", MOM), ("BCTROT-004", BCT)])
def test_a_registered_envelope_ACCEPTS_the_config_it_was_fitted_on(lane, slots):
    a = assess(reg.REGISTERED[lane], {"return_pct": 0.5},
               config_fingerprint=reg.fingerprint(_cfg(slots)),
               live_sessions=13, live_windows=3)
    assert a.state == IN_ENVELOPE, a.reasons


@pytest.mark.parametrize("lane,slots", [("MOMENTUM-002", MOM), ("BCTROT-004", BCT)])
def test_a_registered_envelope_REFUSES_the_OTHER_lane_s_config(lane, slots):
    """The sharpest version of the versioning rule, because these two are the most confusable pair
    in the repo: one is a subclass of the other."""
    other = BCT if slots == MOM else MOM
    with pytest.raises(StaleEnvelope):
        assess(reg.REGISTERED[lane], {"return_pct": 0.5},
               config_fingerprint=reg.fingerprint(_cfg(other)),
               live_sessions=13, live_windows=3)


# -- what the evidence can carry ----------------------------------------------------------------------

@pytest.mark.parametrize("lane", ["MOMENTUM-002", "BCTROT-004"])
def test_NO_p5_OR_p95_IS_REGISTERED(lane):
    """14 independent 13-session windows cannot estimate a 5th percentile. Registering one would put
    a number the evidence does not support at the exact edge where the band decides in/out.

    ABSENT, not merely unused: a distribution that contains a number invites its use.
    """
    for metric, points in reg.REGISTERED[lane].distribution.items():
        assert 5 not in points and 95 not in points, (
            f"{lane}.{metric} registers p5/p95 off 180 autocorrelated overlapping windows")
        assert min(points) >= 10 and max(points) <= 90, points


@pytest.mark.parametrize("lane", ["MOMENTUM-002", "BCTROT-004"])
def test_samples_is_the_INDEPENDENT_window_count(lane):
    """193 tradable sessions / 13 = 14. The 180 overlapping windows are autocorrelated and would
    overstate the evidence 13x. `Envelope.samples` is a claim about evidence, which is why it is a
    declared field rather than an implementation detail."""
    assert reg.REGISTERED[lane].samples == 14, (
        "samples is not the independent count — 180 overlapping windows is not 180 observations")


@pytest.mark.parametrize("lane", ["MOMENTUM-002", "BCTROT-004"])
def test_the_band_is_10_90(lane):
    for metric, band in reg.REGISTERED[lane].bands.items():
        assert band == (10.0, 90.0), f"{lane}.{metric} band is {band}, not the (10, 90) fitted"


@pytest.mark.parametrize("lane", ["MOMENTUM-002", "BCTROT-004"])
def test_a_lane_does_not_act_before_THREE_windows(lane):
    """The first bad month is precisely when a lane most needs to be left alone."""
    assert reg.REGISTERED[lane].min_windows_to_act == 3


# -- the finding that is the point ---------------------------------------------------------------------

def test_BEING_DOWN_IS_INSIDE_BOTH_LANES_ENVELOPES():
    """THE WHOLE CASE FOR THE HOOK, asserted rather than described. 46% and 49% of these lanes'
    13-session windows are NEGATIVE — against TECHIVOL's 38%, which was itself enough to stop a
    lane being blamed for a selection failure it did not have.

    So a 2% loss over 13 sessions is ORDINARY for both, and a drawdown alarm firing on it would be
    firing on measured behaviour. If this test ever goes red because p25 rose above -2%, the lanes
    changed and the bands are stale — which is the fingerprint's job to catch first.
    """
    for lane, slots in (("MOMENTUM-002", MOM), ("BCTROT-004", BCT)):
        a = assess(reg.REGISTERED[lane], {"return_pct": -2.0},
                   config_fingerprint=reg.fingerprint(_cfg(slots)),
                   live_sessions=13, live_windows=3)
        assert a.state == IN_ENVELOPE, f"{lane} calls a 2% loss abnormal: {a.reasons}"
        assert a.acts is False


def test_TURNOVER_IS_ASSESSED_SEPARATELY_FROM_RETURN():
    """TECHIVOL's return was at the 13th percentile — low, but inside. Its TURNOVER was the defect.
    An envelope carrying only `return_pct` can say "bad window" and cannot say which part is wrong,
    which is the sentence that mattered.

    Here: BCTROT delivering MOMENTUM's turnover is out-of-envelope even though the return is fine.
    """
    a = assess(reg.REGISTERED["BCTROT-004"],
               {"return_pct": 0.17, "trades_per_session": 3.9},
               config_fingerprint=reg.fingerprint(_cfg(BCT)),
               live_sessions=13, live_windows=3)
    assert a.state == OUT_OF_ENVELOPE, a.reasons
    assert any("trades_per_session" in r for r in a.reasons), a.reasons
    assert not any("return_pct" in r and "BEYOND" in r for r in a.reasons), (
        "the return was flagged too — the two metrics are not being assessed independently")


def test_WITHOUT_THE_GAP_RULE_BCTROT_TRADES_MORE_FOR_A_LOWER_MEDIAN():
    """The gap-OFF reading: 1.7x the turnover for a lower median, same pool, same exits, only the
    clock differing. Asserted here rather than left in a findings file that can drift — and named
    for its CONFIGURATION, because the companion below shows it does not survive one."""
    m = reg.REGISTERED["MOMENTUM-002"].distribution["trades_per_session"][50]
    b = reg.REGISTERED["BCTROT-004"].distribution["trades_per_session"][50]
    assert b > 1.5 * m, f"BCTROT median turnover {b} is not the measured ~1.7x of MOMENTUM's {m}"
    assert (reg.REGISTERED["BCTROT-004"].distribution["return_pct"][50]
            < reg.REGISTERED["MOMENTUM-002"].distribution["return_pct"][50]), (
        "BCTROT's median return is no longer BELOW MOMENTUM's in the gap-off fit")


def test_THE_GAP_RULE_REVERSES_THE_134_COMPARISON():
    """THE CORRECTION, and the reason #134 needs both numbers in front of it.

    `min_abs_gap_pct` removes 45% of candidates, and BCTROT takes three bites at that middle to
    MOMENTUM's one. At the live 1.5% threshold the turnover premium falls from 1.69x to 1.35x and
    the median return difference CHANGES SIGN. BCTROT's 13-session drawdown percentile also becomes
    the better of the two.

    So the gap-off reading is not robust to a live configuration detail. That is a finding about the
    #134 question rather than an answer to it, and it is asserted so that a later refit which
    quietly restores the original ranking has to argue with a red test.
    """
    m = reg.REGISTERED["MOMENTUM-002+gap"].distribution
    b = reg.REGISTERED["BCTROT-004+gap"].distribution
    ratio = b["trades_per_session"][50] / m["trades_per_session"][50]
    assert 1.2 < ratio < 1.5, f"the turnover premium is {ratio:.2f}x, not the measured ~1.35x"
    assert b["return_pct"][50] >= m["return_pct"][50], (
        "with the gap rule BCTROT's median is no longer at or above MOMENTUM's — the reversal that "
        "makes #134 configuration-dependent has gone")
    assert b["return_pct"][10] > m["return_pct"][10], (
        "BCTROT's 13-session p10 is no longer the better of the two under the gap rule")


def test_A_LANE_RUNNING_THE_GAP_RULE_REFUSES_THE_GAP_OFF_BANDS():
    """The safety property that makes registering both configurations honest rather than confusing.

    The two fits differ by a third in turnover. A lane reading the wrong one would place a perfectly
    normal window far outside its band, or a genuinely broken one comfortably inside it. `fingerprint`
    carries `gap_rule`, so this refuses instead.
    """
    live_gap = reg.fingerprint(MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=8, buffer=5),
        exits=ExitConfig(give_back_frac=0.5),
        execution=ExecutionConfig(decision_slots=BCT, min_abs_gap_pct=reg.LIVE_GAP_PCT)))
    with pytest.raises(StaleEnvelope):
        assess(reg.REGISTERED["BCTROT-004"], {"return_pct": 0.5},
               config_fingerprint=live_gap, live_sessions=13, live_windows=3)
    # and the gap-on envelope accepts it
    a = assess(reg.REGISTERED["BCTROT-004+gap"], {"return_pct": 0.5},
               config_fingerprint=live_gap, live_sessions=13, live_windows=3)
    assert a.state == IN_ENVELOPE, a.reasons


# -- discovered, not listed ----------------------------------------------------------------------------

def test_every_registered_envelope_is_STAMPED():
    """#138: a hand-written list dropped BCTROT because it inherits its handler. Every entry is
    enumerated from the registry rather than named, so a lane added tomorrow is checked here."""
    assert reg.REGISTERED, "the registry is empty — every test above is vacuous"
    for lane, env in reg.REGISTERED.items():
        assert env.config_fingerprint, f"{lane} carries an empty fingerprint"
        assert env.config_fingerprint.startswith("mom:"), env.config_fingerprint
        assert env.distribution and env.bands, lane


def test_the_registry_does_not_leak_UNSTAMPED_module_level_envelopes():
    """`MOMENTUM_002` and `BCTROT_004` are declared with an empty fingerprint and stamped into
    `REGISTERED`. Using the unstamped one would assess against `""` and refuse everything — or, if
    `assess` ever stopped refusing, silently accept everything."""
    assert reg.MOMENTUM_002.config_fingerprint == ""
    with pytest.raises(StaleEnvelope):
        assess(reg.MOMENTUM_002, {"return_pct": 0.0},
               config_fingerprint=reg.fingerprint(_cfg(MOM)),
               live_sessions=13, live_windows=3)


def test_the_bands_live_with_the_STRATEGY_not_in_a_runner():
    """Rule 2 of the market-view protocol, applied to envelopes: the declaration travels with the
    strategy. A runner that carried its own bands would give the lane's behaviour two sources."""
    runners = pathlib.Path(envelope_mod.__file__).parent.parent / "backtesting"
    offenders = [p.name for p in sorted(runners.glob("runner*.py"))
                 if "Envelope(" in p.read_text()]
    assert not offenders, f"a runner declares an envelope of its own: {offenders}"


def test_THE_DEPLOYED_PAIRING_IS_REGISTERED_AND_IS_THE_ASYMMETRIC_ONE():
    """THE THIRD PAIRING, and the only one that describes the running lanes.

    Confirmed from the paper stack 2026-09-11 by three agreeing derivations: MOMENTUM-002 runs ONE
    slot with NO gap filter; BCTROT-004 runs THREE slots WITH the filter at 1.5%. Neither
    both-off nor both-on is deployed.

    Measured on that pairing BCTROT trades **0.69x** MOMENTUM's turnover — LESS, not 1.7x more —
    for -0.27pp of median return and a 0.93pp BETTER drawdown percentile. Every version of "BCTROT
    churns more for less" compared configurations that are not deployed together.

    Asserted here so a later refit that quietly restores the original ranking has to argue with a
    red test, and so the two live keys cannot be renamed without noticing.
    """
    live_mom = reg.REGISTERED["MOMENTUM-002"]          # 1 slot, gap off
    live_bct = reg.REGISTERED["BCTROT-004+gap"]        # 3 slots, gap 1.5%
    ratio = (live_bct.distribution["trades_per_session"][50]
             / live_mom.distribution["trades_per_session"][50])
    assert ratio < 1.0, (
        f"on the DEPLOYED pairing BCTROT is trading {ratio:.2f}x MOMENTUM — the measured value is "
        f"0.69x, i.e. LESS. If this flips, the #134 evidence has changed sign again.")
    assert (live_bct.distribution["return_pct"][10]
            > live_mom.distribution["return_pct"][10]), (
        "BCTROT's 13-session p10 is no longer the better of the deployed pair")
