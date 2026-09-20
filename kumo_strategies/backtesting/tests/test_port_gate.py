"""The acceptance gate for #120 phase 2, and the ablation gate the operator asked for.

2026-09-11: **"it needs detailed testing to not harm the outcomes."** That is the correct
thing to be afraid of, and it is what this file is. Porting a lane onto one engine is worth doing
only if there is something that would SAY SO when it changed the historical numbers.

The organising idea, which is CLAUDE.md's rule stated as an API:

    port / deployment   IDENTICAL IS THE PASS   — differing when they should match is a live defect
    ablation            IDENTICAL IS THE FAIL   — identical when it should differ is a dead
                                                  mechanism

A market view that is declared but never binds produces byte-identical arms. A gate that reports
that as "within tolerance" has inverted its own finding: nothing happened is not the same as
nothing broke. #149 is the live instance — TECHIVOL's 50-day view is declared on the config and
NOTHING READS IT.
"""

from __future__ import annotations

import pytest

from kumo_strategies.backtesting.port_gate import (
    AGREES_UNPROVEN,
    CONDITIONAL,
    FAIL,
    INERT,
    PASS,
    PROVEN,
    STAMP_DERIVED,
    UNKNOWN,
    ConfigField,
    ConfigSheet,
    PortGate,
    SessionCounts,
    Tolerance,
    ViewArm,
    split_view_effect,
    straddles_change,
)


def _sheet(*fields, lane="TECHIVOL-005"):
    return ConfigSheet(lane=lane, fields=fields or (
        ConfigField("slots", ("open+150m",), PROVEN, source="BUILTIN_SLOTS + settings []",
                    sessions_consistent=5),))


def _gate(sheet=None, kind="port"):
    return PortGate(sheet or _sheet(), kind=kind)


# -- a tolerance chosen after the delta is not a gate -------------------------------------------------

def test_DECLARING_AFTER_EVALUATING_RAISES():
    """It never happens as a decision. It happens as a number nudged while staring at a near-miss,
    and by then the gate has already told you the answer you are about to permit."""
    from kumo_strategies.backtesting.port_gate import AlreadyEvaluated

    g = _gate().declare(Tolerance("return_pct", 0.1, "phase 1 reproduced to 0.0000"))
    g.evaluate({"return_pct": 0.05})
    with pytest.raises(AlreadyEvaluated, match="AFTER a delta was seen"):
        g.declare(Tolerance("return_pct", 1.0, "it was close"))


def test_A_TOLERANCE_NEEDS_A_STATED_REASON():
    """"0.1 pp" with no basis is indistinguishable from a number chosen to fit a delta somebody
    had already seen."""
    with pytest.raises(ValueError, match="stated reason"):
        Tolerance("return_pct", 0.1, "  ")


def test_AN_UNBOUNDED_GATE_IS_REFUSED():
    """A gate with no tolerances passes everything, which is worse than no gate because it reports
    a verdict."""
    with pytest.raises(ValueError, match="no tolerances declared"):
        _gate().evaluate({"return_pct": 999.0})


def test_A_METRIC_WITH_NO_DECLARED_TOLERANCE_IS_REFUSED():
    """Judging a metric whose limit was never stated is the same failure as choosing the limit
    afterwards — it just looks tidier."""
    g = _gate().declare(Tolerance("return_pct", 0.1, "phase 1"))
    with pytest.raises(ValueError, match="no tolerance declared"):
        g.evaluate({"return_pct": 0.0, "sharpe": 0.4})


# -- provenance: a delta on an unconfirmed field is not a reproduction --------------------------------

def test_AN_UNKNOWN_FIELD_MAKES_THE_RESULT_CONDITIONAL_NEVER_PASS():
    """THE RULE THE WHOLE SHEET EXISTS FOR. A delta inside tolerance on a field nobody can confirm
    has not been shown to be a reproduction — it has been shown to be SMALL, which is also what a
    wrong config looks like when the two happen to be close.

    And the cheap resolution to an unattributable delta is to tune until it matches, which ports
    the wrong strategy behind a green gate.
    """
    sheet = _sheet(ConfigField("allocated_equity", None, UNKNOWN))
    r = _gate(sheet).declare(Tolerance("return_pct", 0.1, "phase 1")).evaluate({"return_pct": 0.0})
    assert r.verdict == CONDITIONAL
    assert r.conditional_on == ("allocated_equity",)
    assert "CONDITIONAL" in r.summary()


def test_A_COMPLETE_SHEET_CAN_PASS():
    r = _gate().declare(Tolerance("return_pct", 0.1, "phase 1")).evaluate({"return_pct": 0.0})
    assert r.verdict == PASS and r.conditional_on == ()


def test_A_BREACH_IS_A_FAIL_WHATEVER_THE_PROVENANCE():
    """CONDITIONAL is weaker than PASS, not weaker than FAIL. An unconfirmed config does not excuse
    a delta outside the limit."""
    sheet = _sheet(ConfigField("allocated_equity", None, UNKNOWN))
    r = _gate(sheet).declare(Tolerance("return_pct", 0.1, "phase 1")).evaluate({"return_pct": 5.0})
    assert r.verdict == FAIL


def test_A_SINGLE_JOURNAL_STAMP_IS_NOT_TRUSTWORTHY():
    """THE `open+215m` CASE. One row, 2026-08-24 — a DELAYED FIRING recorded faithfully and read as
    a setting. A stamp proves what the runner USED at that instant, not what the config SAYS."""
    one_row = ConfigField("slots", ("open+215m",), STAMP_DERIVED, source="journal 2026-08-24",
                          sessions_consistent=1)
    assert one_row.trustworthy is False


def test_A_STAMP_THAT_HELD_ACROSS_SESSIONS_IS_TRUSTWORTHY():
    """THE COUNT, NOT THE CATEGORY NAME. BCTROT's three slots held 3 of 3 sessions; MOMENTUM's one
    held 5 of 5. Same category as the retracted single row, different evidence."""
    held = ConfigField("slots", ("open+5m", "open+150m", "close-20m"), STAMP_DERIVED,
                       source="journal 09-08..09-10", sessions_consistent=3)
    assert held.trustworthy is True


def test_AGREEMENT_BETWEEN_TWO_NON_AUTHORITATIVE_SOURCES_IS_NOT_PROOF():
    """`min_abs_gap_pct` was inherited from the sibling that "obviously" had the same value, and
    that inheritance flipped BCTROT's sign. Agreement is evidence; it is not proof."""
    f = ConfigField("min_abs_gap_pct", 0.015, AGREES_UNPROVEN, source="repo default == observed")
    assert f.trustworthy is False


def test_A_KNOWN_VALUE_MUST_NAME_ITS_SOURCE():
    """'proven' with no source is an assertion, and assertions about live config are what moved
    three published numbers in one day."""
    with pytest.raises(ValueError, match="name where it came from"):
        ConfigField("slots", ("open+5m",), PROVEN)


def test_AN_UNRECOGNISED_PROVENANCE_IS_REFUSED_NOT_DEFAULTED():
    with pytest.raises(ValueError, match="not one of"):
        ConfigField("slots", (), "probably", source="a feeling")


# -- the ablation gate: identical is the FAILURE -------------------------------------------------------

def test_IDENTICAL_ARMS_ARE_INERT_NOT_A_PASS():
    """THE INVERSION, and the reason this is one class with a declared direction rather than two
    tools. A view declared and never bound produces byte-identical arms. Reporting that as "within
    tolerance" records a DEAD MECHANISM as a SAFE one.

    #149 is the live instance: TECHIVOL's 50-day view is declared on the config and `grep
    market_view` across `runtime/` returns zero hits.
    """
    g = PortGate(_sheet(), kind="ablation").declare(
        Tolerance("return_pct", 1.0, "an effect smaller than this is not worth the complexity"))
    r = g.evaluate({"return_pct": 0.0})
    assert r.verdict == INERT, "identical arms passed an ablation gate"
    assert "never bound" in r.breaches[0]


def test_THE_SAME_IDENTICAL_RESULT_IS_A_PASS_FOR_A_PORT_GATE():
    """VERIFY BY DISAGREEMENT, as a test: the same input, opposite verdicts, because the two gates
    ask opposite questions. If these ever agree, one of them has lost its direction."""
    port = _gate(kind="port").declare(Tolerance("return_pct", 0.1, "phase 1 reproduced to 0.0000"))
    abl = PortGate(_sheet(), kind="ablation").declare(Tolerance("return_pct", 1.0, "materiality"))
    assert port.evaluate({"return_pct": 0.0}).verdict == PASS
    assert abl.evaluate({"return_pct": 0.0}).verdict == INERT


def test_AN_ABLATION_WITH_A_REAL_EFFECT_IS_JUDGED_ON_ITS_TOLERANCE():
    g = PortGate(_sheet(), kind="ablation").declare(Tolerance("return_pct", 1.0, "materiality"))
    assert g.evaluate({"return_pct": -8.0}).verdict == FAIL


# -- the two halves of a view are different bets -------------------------------------------------------

def test_BLOCKED_ENTRIES_AND_LIQUIDATION_ARE_REPORTED_SEPARATELY():
    """A net figure hides one behind the other AND THEY CAN CANCEL. Blocked entries during a
    drawdown cost the RECOVERY — which for momentum is where most of the return lives. A
    liquidation realises losses and then pays to re-enter. One number for both records a mechanism
    that helps on one axis and hurts on the other as neutral."""
    split = split_view_effect({"blocked_return_pct": +4.0, "liquidated_return_pct": -4.0})
    assert split["blocked"]["blocked_return_pct"] == 4.0
    assert split["liquidated"]["liquidated_return_pct"] == -4.0
    assert sum(split["blocked"].values()) + sum(split["liquidated"].values()) == 0.0, (
        "the fixture must be one that CANCELS, or it does not exercise the reason for splitting")


def test_A_MISSING_HALF_IS_AN_EMPTY_GROUP_NOT_A_FOLDED_TOTAL():
    """A caller that measured only one half must show an empty other half rather than have it
    silently absorbed — which would read as "liquidation had no effect"."""
    split = split_view_effect({"blocked_return_pct": 4.0})
    assert split["liquidated"] == {}


# -- the window is not inheritable ----------------------------------------------------------------------

def test_A_VIEW_ARM_HAS_NO_DEFAULT_WINDOW():
    """50 is TECHIVOL's fitted answer on TECHIVOL's data. A lane that inherits it has inherited a
    number nobody measured for it — the same error as inheriting `min_abs_gap_pct` from a sibling,
    which is the one that flipped BCTROT's sign."""
    import inspect
    params = inspect.signature(ViewArm).parameters
    assert params["window"].default is inspect.Parameter.empty, (
        "window has a default — a lane can silently inherit a constant fitted elsewhere")
    assert params["signal"].default is inspect.Parameter.empty


def test_A_VIEW_THAT_ACTS_MUST_SAY_WHAT_IT_DOES():
    with pytest.raises(ValueError, match="must say what it does"):
        ViewArm(lane="QC345-003", signal="index_vs_ma", window=50, action=None, acts=True)


def test_A_SHADOW_ARM_NEEDS_NO_ACTION():
    """OBSERVE-ONLY IS THE FIRST WIRING, not a detour: evaluate, fire the event, render the label,
    GATE NOTHING — then compare what the view WOULD have done against what the lane did. The
    notification half is already built and tested; the loop hooks are not wired at all, so shadow
    is the natural first step. CRSISHORT's `shadow_only` refusal is the in-repo precedent."""
    arm = ViewArm(lane="QC345-003", signal="index_vs_ma", window=50, action=None, acts=False)
    assert "shadow" in arm.label


def test_DWELL_IS_EXPRESSIBLE_AND_DEFENDED():
    """dwell=1 means one session flips the state, and a whipsaw month is the scenario that turns a
    protective mechanism into a fee generator. It must be sweepable, and it must not be zero."""
    assert ViewArm("L", "index_vs_ma", 50, "exit_only", dwell=5).dwell == 5
    with pytest.raises(ValueError, match="dwell"):
        ViewArm("L", "index_vs_ma", 50, "exit_only", dwell=0)


def test_THE_CONTROL_ARM_IS_EXPRESSIBLE():
    """An ablation needs a no-view arm, and it must not be forced to invent a window for it."""
    assert ViewArm(lane="L", signal=None, window=0, action=None).label == "no view"


# -- three states in the result --------------------------------------------------------------------------

def test_AN_EXIT_ONLY_MONTH_NEVER_RENDERS_AS_EMPTY():
    """QC345 has 20 of 21 such sessions a month. "Evaluated, nothing to do" is not "not evaluated",
    and a surface that shows the first as empty reads as a broken lane. Cockpit spent #859 and #884
    enforcing this on the display plane; the backtest surface must not reintroduce it."""
    c = SessionCounts(in_panel=21, evaluated=21, decided=1, exit_only=20)
    rendered = c.render()
    assert "decided 1 of 21 evaluated" in rendered
    assert "20 exit-only" in rendered
    assert rendered.strip()


def test_A_LANE_THAT_DECIDED_NOTHING_STILL_RENDERS_A_RESULT():
    c = SessionCounts(in_panel=30, evaluated=21, decided=0, exit_only=21)
    assert "decided 0 of 21 evaluated" in c.render()


def test_COUNTS_THAT_CANNOT_BE_TRUE_ARE_REFUSED():
    """Warm-up withholding means evaluated <= in_panel always; decided and exit_only partition the
    evaluated ones. A surface fed impossible counts renders a plausible lie."""
    with pytest.raises(ValueError):
        SessionCounts(in_panel=10, evaluated=20, decided=0)
    with pytest.raises(ValueError):
        SessionCounts(in_panel=20, evaluated=10, decided=8, exit_only=8)


# -- the delta is the number, with its sign ---------------------------------------------------------------

def test_THE_SUMMARY_CARRIES_THE_SIGNED_DELTA():
    """"within tolerance" hides whether the port made the lane look BETTER or WORSE, and only one
    of those is a reproduction."""
    r = _gate().declare(Tolerance("return_pct", 1.0, "phase 1")).evaluate({"return_pct": -0.42})
    assert "-0.4200" in r.summary()


def test_A_PORT_GATE_AND_A_DEPLOYMENT_GATE_ARE_LABELLED_DIFFERENTLY():
    """They answer different questions and a passing port gate must never be reported as the
    deployment answer."""
    port = _gate(kind="port").declare(Tolerance("r", 1.0, "x")).evaluate({"r": 0.0})
    dep = _gate(kind="deployment").declare(Tolerance("r", 1.0, "x")).evaluate({"r": 0.0})
    assert "port gate" in port.summary() and "deployment gate" in dep.summary()


def test_AN_UNKNOWN_GATE_KIND_IS_REFUSED():
    with pytest.raises(ValueError, match="kind must be one of"):
        PortGate(_sheet(), kind="vibes")


# -- a window that straddles a change event measures two configurations ------------------------------------

def test_A_WINDOW_SPANNING_A_CHANGE_EVENT_IS_FLAGGED():
    """BCTROT is the worked example: cockpit 4e5778a (2026-09-04) added the third decision slot AND
    `min_abs_gap_pct=0.015` in ONE commit. Before it, two slots and no gap filter. A window
    spanning that date measures both halves and reports one number — and they are not separable
    after the fact, because they never varied independently."""
    assert straddles_change("2025-12-01", "2026-09-08", ("2026-09-04",)) == ("2026-09-04",)


def test_A_WINDOW_CLEAR_OF_CHANGE_EVENTS_IS_CLEAN():
    assert straddles_change("2025-12-01", "2026-08-30", ("2026-09-04",)) == ()
