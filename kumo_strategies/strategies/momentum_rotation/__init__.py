"""Momentum rotation — hold a few names showing strength, rotate as strength moves.

Configurable by design: the candidate pool is a plug-in, because measurement showed the pool IS the
edge and the ranking rule is worthless without a good one (issue #15).
"""

from .config import MomentumRotationConfig
from .engine import Decision, apply_gates, decide, score_panel

__all__ = ["MomentumRotationConfig", "Decision", "apply_gates", "score_panel", "decide"]
