"""A session that fails is LOGGED, not lost in the logging call (ks#233, cockpit 2026-09-14 handoff).

Measured on ibkr-paper at 13:20:00.543Z on 2026-09-14, CRSISHORT-006's first live slot:
`ERROR CRSISHORT-006: session WAS NOT WRITTEN — TypeError: error() got an unexpected keyword argument
'exc_info'`. The decision ran; the record of its failure did not land, because Nautilus's `Logger.error`
takes ONE message string and the except-branch passed Python logging's `exc_info=True`. The test double
in every other file accepts `**k`, which is exactly why this went green for a month: a double must
reject what production rejects.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd

from kumo_strategies.strategies.crsi_short.nautilus import CrsiShortStrategy
from kumo_strategies.strategies.template.nautilus import TemplateRotationStrategy


class _NautilusShapedLog:
    """`nautilus_trader.common.component.Logger`: `error(msg: str, color=..., ...)` — no `exc_info`."""

    def __init__(self):
        self.errors: list[str] = []

    def info(self, msg, *a):
        pass

    def warning(self, msg, *a):
        pass

    def error(self, msg, color=None):
        self.errors.append(msg)


class _Runner:
    async def run(self, panel, session, *, slot=None, jobs=None, opens=None):
        raise RuntimeError("the runner blew up mid-session")


def _crsi_host():
    h = SimpleNamespace()
    h.id = "CRSISHORT-001"
    h.log = _NautilusShapedLog()
    h._runner = _Runner()
    h._shadow_only = False
    h._held = set()
    h._cfg = None
    h.cache = SimpleNamespace(positions_open=lambda **kw: [])
    h.POSITION_SIDE = CrsiShortStrategy.POSITION_SIDE
    h._session_coro = CrsiShortStrategy._session_coro.__get__(h)
    return h


def test_CRSISHORT_logs_a_failed_session_through_a_logger_that_rejects_exc_info():
    h = _crsi_host()
    panel = pd.DataFrame({"ticker": [], "date": [], "close": []})
    asyncio.run(h._session_coro(pd.Timestamp("2026-09-14"), panel, "open-10m"))
    assert len(h.log.errors) == 1, h.log.errors
    msg = h.log.errors[0]
    assert "session 2026-09-14 failed" in msg and "RuntimeError: the runner blew up" in msg
    assert "Traceback" in msg, "the traceback must be IN the message — the logger has no exc_info"


def test_the_rotation_template_logs_a_failed_session_the_same_way():
    h = SimpleNamespace()
    h.id = "ROT-001"
    h.log = _NautilusShapedLog()
    h._runner = _Runner()
    h._session_coro = TemplateRotationStrategy._session_coro.__get__(h)
    asyncio.run(h._session_coro(pd.Timestamp("2026-09-14"), pd.DataFrame(), "open+5m"))
    assert len(h.log.errors) == 1 and "Traceback" in h.log.errors[0], h.log.errors
