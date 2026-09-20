"""Cluster-level exit — fire on the cluster WEAKENING, never on the cluster being LARGE (#30).

The distinction is the entire design and it is easy to implement backwards. #25 measured where the
money is: the wildest volatility quartile carried +90.4% of P&L, so concentration is the edge. A rule
that fires on cluster SIZE caps exactly what pays.

The failure it targets, from MOMENTUM-002's live journal: on 2026-08-13 four precious-metals miners
exited together, and on 2026-08-14 all four were bought back. The ranking treats them as four
independent names, each still inside the buffer until it individually drops out. They are one bet.
"""

from __future__ import annotations

import pandas as pd

from kumo_strategies.strategies.momentum_rotation.config import (
    MomentumRotationConfig, PortfolioConfig)
from kumo_strategies.strategies.momentum_rotation.engine import cluster_exits

MINERS = ["AEM", "CGAU", "FSM", "WPM"]
LONE = "VFLO"


def _corr(rho_within: float = 0.9, rho_across: float = 0.05) -> pd.DataFrame:
    names = MINERS + [LONE]
    m = pd.DataFrame(rho_across, index=names, columns=names)
    for a in MINERS:
        for b in MINERS:
            m.loc[a, b] = 1.0 if a == b else rho_within
    m.loc[LONE, LONE] = 1.0
    return m


def _cfg(floor=None, corr_thr=0.7):
    return MomentumRotationConfig(portfolio=PortfolioConfig(
        n_hold=8, buffer=5, cluster_exit_score=floor, cluster_corr=corr_thr))


def test_disabled_by_default():
    ranked = pd.Series({s: -5.0 for s in MINERS + [LONE]})
    assert cluster_exits(set(MINERS), ranked, _corr(), _cfg()) == set()


def test_a_weak_cluster_exits_as_a_GROUP():
    """The 08-13 case. All four are weak together, so all four go at once rather than one a session."""
    ranked = pd.Series({"AEM": -0.5, "CGAU": -0.4, "FSM": -0.6, "WPM": -0.3, LONE: 2.0})
    got = cluster_exits(set(MINERS + [LONE]), ranked, _corr(), _cfg(floor=0.0))
    assert got == set(MINERS), f"expected the whole cluster, got {got}"
    assert LONE not in got, "an uncorrelated strong name was dragged out with the cluster"


def test_a_STRONG_cluster_is_left_alone_however_large():
    """The constraint from #25. Four correlated names all working is the trade, not a problem."""
    ranked = pd.Series({s: 2.5 for s in MINERS} | {LONE: 2.0})
    assert cluster_exits(set(MINERS + [LONE]), ranked, _corr(), _cfg(floor=0.0)) == set()


def test_one_weak_name_inside_a_strong_cluster_survives():
    """The point of judging the CLUSTER. A single laggard among working peers is noise; the ranking
    and the buffer already exist to avoid churning on it."""
    ranked = pd.Series({"AEM": -0.5, "CGAU": 2.5, "FSM": 2.6, "WPM": 2.4, LONE: 2.0})
    assert cluster_exits(set(MINERS + [LONE]), ranked, _corr(), _cfg(floor=0.0)) == set()


def test_a_weak_UNCORRELATED_name_is_judged_on_itself():
    """A singleton cluster is the name alone, so the rule reduces to its own score and adds nothing
    the ranking does not already do."""
    ranked = pd.Series({s: 2.5 for s in MINERS} | {LONE: -1.0})
    assert cluster_exits({LONE, *MINERS}, ranked, _corr(), _cfg(floor=0.0)) == {LONE}


def test_the_correlation_threshold_defines_membership():
    """Below the threshold the miners are five separate bets and each is judged alone, so only the
    ones individually below the floor exit."""
    ranked = pd.Series({"AEM": -0.5, "CGAU": 2.5, "FSM": 2.6, "WPM": 2.4, LONE: 2.0})
    got = cluster_exits(set(MINERS + [LONE]), ranked, _corr(rho_within=0.5), _cfg(floor=0.0))
    assert got == {"AEM"}, f"membership ignored the threshold: {got}"


def test_no_correlation_matrix_means_no_exits_rather_than_silent_full_book():
    """The #26 shape. Without the input the rule must do nothing visible, and `needs_panel_stats`
    is what guarantees a driver actually supplies it."""
    ranked = pd.Series({s: -5.0 for s in MINERS})
    assert cluster_exits(set(MINERS), ranked, None, _cfg(floor=0.0)) == set()


def test_needs_panel_stats_reports_true_so_drivers_supply_corr():
    from kumo_strategies.strategies.momentum_rotation.engine import needs_panel_stats
    assert needs_panel_stats(_cfg(floor=0.0)) is True
    assert needs_panel_stats(_cfg()) is False
