"""MOMENTUM-002's and BCTROT-004's pre-registered envelopes, fitted at their OWN fill hours (#147).

Both lanes are the same strategy on different clocks — same ledger-provider pool, same engine, same exits.
So they are declared together, and a difference between their two distributions is attributable to
the schedule and to nothing else. `bctrot_rotation.py` makes the same argument for the live
comparison: "Changing the schedule and the exit rules together would make the live comparison against
MOMENTUM-002 useless: two changes, one number."

MEASURED 2026-09-11, `research/envelopes/FINDINGS.md`. One continuous warm run per lane over the
book's own lifetime, windows cut from the single active-equity curve.

WHY THE BANDS ARE (10, 90) AND NOT (5, 95)
------------------------------------------
`LedgerBook.COVERAGE_START` is 2025-12-01 and it is a DELIBERATE REFUSAL, not a gap to route around:
before it, community coverage recovered only partially, and answering anyway would quietly invent
book membership. So these lanes have 193 tradable sessions — **14 independent 13-session windows** —
and rebuilding the price panel from 300 to 568 sessions did not move that count by one, because the
SOURCE is the binding constraint and not the prices.

A 5th percentile cannot be estimated from 14 observations. Reading one off the 180 OVERLAPPING
windows would produce a number that looks precise, sits at the exact edge where the band decides
in/out, and overstates the evidence by 13x. So p5/p95 are not registered at all — not registered and
unused, but absent, because a distribution that contains a number invites its use.

`samples` carries the INDEPENDENT count for the same reason. It is a claim about evidence.

TWO CONFIGURATIONS ARE REGISTERED, WITH AND WITHOUT THE GAP RULE
----------------------------------------------------------------
`min_abs_gap_pct` declines an entry whose overnight gap sits inside a dead band. An earlier version
of this module registered only the gap-OFF fit and claimed the runner could not express the rule.
It can: `run_sessions` refuses the gap rule only with `fill="moo"`, these lanes run `next_open`, and
the gap filter applies at every slot. The rule was available and was not set.

It is not a detail. The filter removes 45% of candidates, and BCTROT takes three bites at that
middle to MOMENTUM's one:

    gap        MOM fills   BCT fills   turnover ratio   BCTROT median vs MOMENTUM
    off              774        1302        1.69x             -0.62pp
    1.25%            451         656        1.42x             -0.20pp
    1.50% (live)     397         540        1.35x             +0.09pp
    1.75%            346         455        1.29x             +0.19pp

So "BCTROT trades 1.7x the turnover for a lower median" is a GAP-OFF result, and at the live
threshold the return difference CHANGES SIGN.

AND THE DEPLOYED LANES ARE ASYMMETRIC, WHICH IS A THIRD PAIRING AGAIN. Confirmed from the running
paper stack 2026-09-11, three derivations agreeing (cockpit's `BUILTIN_SLOTS`, `/settings/strategies`
showing no override, and the journal's decision-row count per session):

    MOMENTUM-002   ONE slot  (open+5m)                      min_abs_gap_pct = None
    BCTROT-004     THREE slots (open+5m, +150m, close-20m)  min_abs_gap_pct = 0.015

Neither of the two pairings measured above is the deployed one. The deployed pair is MOMENTUM
gap-OFF against BCTROT gap-ON, and it REVERSES the finding outright:

    MOMENTUM-002  1 slot,  gap off    3.92 fills/session   median +0.78%   p10 -4.55%
    BCTROT-004    3 slots, gap 1.5%   2.69 fills/session   median +0.51%   p10 -3.62%

    turnover 0.69x -- BCTROT trades LESS than MOMENTUM, not 1.7x more
    median -0.27pp, and its 13-session drawdown percentile is 0.93pp BETTER

Three slots with a gap filter is LOWER turnover than one slot without one. Every version of "BCTROT
churns more for less" was an artefact of comparing configurations that are not deployed together.

The two live configurations are `REGISTERED["MOMENTUM-002"]` and `REGISTERED["BCTROT-004+gap"]`.
The other two are kept because a fingerprint cannot describe a lane it does not match, and having
the counterfactual registered is what made this visible at all.

ONE WINDOW CAVEAT, STATED: cockpit commit 4e5778a (2026-09-04) added BCTROT's third slot AND the
gap filter in one change. The fit window is 2025-12-01 to 2026-09-08, so it models that config as
though it had always run; the LANE has run it for four sessions. That is the right counterfactual
for an envelope — it describes a configuration, not a history — and it is exactly why
`min_live_sessions=13` will answer UNKNOWN for a while rather than pretending otherwise.
"""

from __future__ import annotations

import hashlib
import json

from kumo_strategies.strategies.envelope import Envelope
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

__all__ = ["fingerprint", "MOMENTUM_002", "BCTROT_004", "REGISTERED"]


def fingerprint(cfg: MomentumRotationConfig) -> str:
    """A hash of the behaviour-bearing config, so a changed lane cannot reuse an old distribution.

    COMPUTED, NEVER HAND-WRITTEN. A version string typed next to an envelope is a label that agrees
    with the config because a human said so, which is this package's recurring defect — a knob that
    matches its configured value because nothing recomputed it. `Envelope`'s VERSIONED rule is only
    real if the fingerprint is derived from the thing it claims to describe.

    THE GAP RULE IS READ OFF `cfg`, NOT PASSED IN, and it was a separate argument until a mutation
    proved that argument dead: removing `min_abs_gap_pct` from the fitted config changed no
    fingerprint and broke no test, because the flag was supplied by hand alongside it. Two
    derivations of one fact, free to disagree — the config saying "no gap" while the fingerprint said
    "gap". Exactly the shape this package keeps finding, and it was in the function whose job is to
    stop a stale claim being trusted.

    One derivation now: whether the gap rule applies IS whether the config sets it.
    """
    payload = {
        "slots": list(cfg.execution.decision_slots or ()),
        "fill": cfg.execution.fill,
        "n_hold": cfg.portfolio.n_hold,
        "buffer": cfg.portfolio.buffer,
        "give_back_frac": cfg.exits.give_back_frac,
        "off_peak_pct": cfg.exits.off_peak_pct,
        "take_profit_atr": cfg.exits.take_profit_atr,
        "cluster_exit_score": cfg.portfolio.cluster_exit_score,
        "min_abs_gap_pct": cfg.execution.min_abs_gap_pct,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "mom:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


#: 193 tradable sessions, 2025-12-01 -> 2026-09-08. 46% of 13-session windows are NEGATIVE — the
#: number this whole hook exists to publish, against TECHIVOL's 38%. A drawdown alarm on this lane
#: fires routinely on behaviour it was measured to have.
MOMENTUM_002 = Envelope(
    config_fingerprint="",          # filled by `_register`, from the lane's actual config
    horizon_sessions=13,
    samples=14,                     # INDEPENDENT windows, not the 180 overlapping ones
    distribution={
        "return_pct": {10: -4.55, 25: -2.71, 50: 0.78, 75: 3.26, 90: 6.38},
        "trades_per_session": {10: 3.00, 25: 3.37, 50: 3.92, 75: 4.62, 90: 5.23},
    },
    bands={"return_pct": (10.0, 90.0), "trades_per_session": (10.0, 90.0)},
    min_live_sessions=13,
    min_windows_to_act=3,
)

#: Same pool, same exits, three slots instead of one. 49% of 13-session windows negative, and
#: **1.7x MOMENTUM's turnover for a lower median** (6.62 vs 3.92 fills/session, +0.17% vs +0.78%).
#: Not a verdict on 14 independent windows — it is the live comparison to watch, and it is evidence
#: on #134 ("fully separate BCTROT from MOMENTUM"), which is the question it actually informs.
BCTROT_004 = Envelope(
    config_fingerprint="",
    horizon_sessions=13,
    samples=14,
    distribution={
        "return_pct": {10: -5.23, 25: -2.40, 50: 0.17, 75: 3.17, 90: 6.61},
        "trades_per_session": {10: 4.92, 25: 5.67, 50: 6.62, 75: 7.85, 90: 8.62},
    },
    bands={"return_pct": (10.0, 90.0), "trades_per_session": (10.0, 90.0)},
    min_live_sessions=13,
    min_windows_to_act=3,
)


def _register(env: Envelope, cfg: MomentumRotationConfig) -> Envelope:
    """Stamp the envelope with the fingerprint of the config it was fitted on.

    `Envelope` is frozen, so this REPLACES rather than mutates — a caller cannot widen a band in
    place, and cannot quietly re-point an envelope at a different config either.
    """
    return Envelope(
        config_fingerprint=fingerprint(cfg),
        horizon_sessions=env.horizon_sessions,
        samples=env.samples,
        distribution=env.distribution,
        bands=env.bands,
        min_live_sessions=env.min_live_sessions,
        min_windows_to_act=env.min_windows_to_act,
    )


#: The configs these were FITTED ON, spelled out here rather than imported from a live module.
#: If a lane's deployed config drifts from this, the fingerprint stops matching and `assess` refuses
#: — which is the entire point of the VERSIONED rule and the reason the fit parameters are written
#: down where the bands are, not somewhere a reader has to trust.
_FITTED_ON = {
    "MOMENTUM-002": ("open+5m",),
    "BCTROT-004": ("open+5m", "open+150m", "close-20m"),
}


def _fitted_cfg(slots: tuple[str, ...]) -> MomentumRotationConfig:
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExecutionConfig, ExitConfig, PortfolioConfig)
    return MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=8, buffer=5),
        exits=ExitConfig(give_back_frac=0.5),
        execution=ExecutionConfig(decision_slots=slots),
    )


#: The same two lanes WITH the 1.5% gap dead band — the live threshold. Turnover falls by roughly a
#: third on both, and BCTROT's 13-session drawdown percentile becomes the BETTER of the two
#: (-3.62% against MOMENTUM's -4.58%), which is the reverse of the gap-off reading.
MOMENTUM_002_GAP = Envelope(
    config_fingerprint="",
    horizon_sessions=13,
    samples=14,
    distribution={
        "return_pct": {10: -4.58, 25: -1.83, 50: 0.42, 75: 2.58, 90: 4.65},
        "trades_per_session": {10: 1.31, 25: 1.69, 50: 2.00, 75: 2.63, 90: 3.00},
    },
    bands={"return_pct": (10.0, 90.0), "trades_per_session": (10.0, 90.0)},
    min_live_sessions=13,
    min_windows_to_act=3,
)

BCTROT_004_GAP = Envelope(
    config_fingerprint="",
    horizon_sessions=13,
    samples=14,
    distribution={
        "return_pct": {10: -3.62, 25: -1.40, 50: 0.51, 75: 2.83, 90: 5.55},
        "trades_per_session": {10: 1.69, 25: 2.08, 50: 2.69, 75: 3.54, 90: 4.08},
    },
    bands={"return_pct": (10.0, 90.0), "trades_per_session": (10.0, 90.0)},
    min_live_sessions=13,
    min_windows_to_act=3,
)

#: The live gap threshold. 1.5% was chosen IN SAMPLE (`FINDINGS-gap-entry.md` says so plainly), which
#: is why the arms span 1.25-1.75% and the ranking is reported as stable across all three rather than
#: asserted at one point.
LIVE_GAP_PCT = 0.015


def _register_gap(env: Envelope, cfg: MomentumRotationConfig) -> Envelope:
    """Identical to `_register` now that the gap rule is read off the config. Kept as a separate
    name only because the two call sites read differently at a glance; it takes no flag, so the two
    cannot drift."""
    return Envelope(
        config_fingerprint=fingerprint(cfg),
        horizon_sessions=env.horizon_sessions, samples=env.samples,
        distribution=env.distribution, bands=env.bands,
        min_live_sessions=env.min_live_sessions, min_windows_to_act=env.min_windows_to_act,
    )


def _fitted_gap_cfg(slots: tuple[str, ...]) -> MomentumRotationConfig:
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExecutionConfig, ExitConfig, PortfolioConfig)
    return MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=8, buffer=5),
        exits=ExitConfig(give_back_frac=0.5),
        execution=ExecutionConfig(decision_slots=slots, min_abs_gap_pct=LIVE_GAP_PCT),
    )


REGISTERED: dict[str, Envelope] = {
    "MOMENTUM-002": _register(MOMENTUM_002, _fitted_cfg(_FITTED_ON["MOMENTUM-002"])),
    "BCTROT-004": _register(BCTROT_004, _fitted_cfg(_FITTED_ON["BCTROT-004"])),
    "MOMENTUM-002+gap": _register_gap(MOMENTUM_002_GAP,
                                      _fitted_gap_cfg(_FITTED_ON["MOMENTUM-002"])),
    "BCTROT-004+gap": _register_gap(BCTROT_004_GAP, _fitted_gap_cfg(_FITTED_ON["BCTROT-004"])),
}
