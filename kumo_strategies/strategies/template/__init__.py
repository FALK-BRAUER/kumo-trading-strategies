"""TEMPLATE strategy — copy this directory to start a new lane.

See `README.md` for the checklist. The pure layer is importable without Nautilus; the runtime adapter
lives beside it in `nautilus.py`, the session gateway in `runner.py` (ks#211).
"""

from kumo_strategies.strategies.template.config import TemplateConfig
from kumo_strategies.strategies.template.engine import (
    Decision, build_feature_panel, decide, rebalance_dates)

__all__ = ["TemplateConfig", "Decision", "build_feature_panel", "decide", "rebalance_dates"]
