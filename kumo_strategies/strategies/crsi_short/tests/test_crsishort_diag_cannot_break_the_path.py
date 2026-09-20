"""CRSISHORT's diagnostics cannot take the path they observe down (#247).

312d904 (deployed to ibkr-paper) logged `self._slots`, `self._min_warm`, `self._ingested_bars` and a
`_panel_diag` method inline on `on_start`, the bar ingest and the session alert. On any host that
lacks one of them the AttributeError took the SESSION down — eleven tests red on the branch's own
base, and live the same shape reads as "session failed" for a log line. The diagnostics are now
module functions that read by name with a default and swallow their own failures. Observation is a
mechanism, not a try/except at each site — and never a raise into the caller.
"""

from __future__ import annotations

import pandas as pd

from kumo_strategies.strategies.crsi_short.nautilus import _attr, _diag, _panel_diag, _ready_count


class _Log:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def info(self, m): self.lines.append(("info", m))
    def error(self, m): self.lines.append(("error", m))
    def debug(self, m): self.lines.append(("debug", m))


class _Bare:
    """A host with a log and an id and NOTHING else — the narrowest double the suite binds to."""
    id = "CRSISHORT-006"

    def __init__(self):
        self.log = _Log()


def test_a_message_that_cannot_be_built_is_logged_at_debug_and_does_not_raise():
    h = _Bare()
    _diag(h, lambda: f"slots={h._slots}")          # AttributeError inside the builder
    assert h.log.lines == [("debug", "CRSISHORT-006: CRSISHORT-DIAG line unavailable: AttributeError: "
                                      "'_Bare' object has no attribute '_slots'")]


def test_a_message_that_builds_is_logged_at_the_level_asked():
    h = _Bare()
    _diag(h, lambda: "hello", level="error")
    _diag(h, "plain")
    assert h.log.lines == [("error", "CRSISHORT-006: CRSISHORT-DIAG hello"), ("info", "CRSISHORT-006: CRSISHORT-DIAG plain")]


def test_a_host_without_a_log_is_silently_fine():
    class _NoLog:
        pass
    _diag(_NoLog(), lambda: 1 / 0)                 # neither the builder nor the missing log raises


def test_attributes_read_by_name_with_a_default_and_readiness_without_bars():
    h = _Bare()
    assert _attr(h, "_min_warm") == "?" and _attr(h, "_min_warm", None) is None
    assert _ready_count(h) == -1
    h._bars = {"A": [1, 2, 3], "B": [1]}
    h._need = 2
    assert _ready_count(h) == 1


def test_panel_diag_on_a_bare_host_describes_the_panel_and_marks_the_unknowns():
    h = _Bare()
    panel = pd.DataFrame({"date": pd.to_datetime(["2026-09-12", "2026-09-14", "2026-09-14"]),
                          "ticker": ["A", "A", "B"]})
    line = _panel_diag(h, panel, pd.Timestamp("2026-09-14"))
    assert line == "rows=3 symbols=2 ready=-1/? need=? oldest=2026-09-12 newest=2026-09-14 session_rows=2"
    assert _panel_diag(h, pd.DataFrame()) == "panel empty"
