"""SMHGLD claims follow terminal order events, never local submit acceptance (#829).

The cockpit runner can accept an order locally while the venue later rejects it, or while restart
replay only proves one leg filled. The adapter therefore syncs the claim from the lane's actual
Nautilus cache quantity on every terminal event.
"""

from __future__ import annotations

from types import SimpleNamespace


class _Host:
    """Real SMHGLD order-event methods bound to a narrow host."""

    from kumo_strategies.strategies.smhgld_sleeve.nautilus import SmhGldSleeveStrategy as _S

    on_order_filled = _S.on_order_filled
    on_order_rejected = _S.on_order_rejected
    on_order_denied = _S.on_order_denied
    _forget = _S._forget
    _name_a_refused_close = _S._name_a_refused_close
    _claim_qty_after_event = _S._claim_qty_after_event
    POSITION_SIDE = _S.POSITION_SIDE
    _CLOSING_TIF = _S._CLOSING_TIF
    id = "SMHGLD-903"

    del _S


class _Runner:
    def __init__(self):
        self.synced = []

    async def sync_claim(self, symbol, qty, px):
        self.synced.append((symbol, qty, px))


def _drain(coro):
    try:
        coro.send(None)
    except StopIteration:
        pass


def _lane(*, qty: int, held=(), pending=None):
    lane = _Host()
    runner = _Runner()
    lane._held = set(held)
    lane._pending = dict(pending or {})
    lane._runner = runner
    lane._loop = object()
    lane.said = []
    lane.log = SimpleNamespace(
        warning=lambda m="", *a, **k: lane.said.append(("warning", str(m))),
        error=lambda m="", *a, **k: lane.said.append(("error", str(m))),
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    lane.cache = SimpleNamespace(
        positions_open=lambda **kw: (
            [SimpleNamespace(signed_qty=qty)] if qty != 0 else []
        )
    )
    lane.fire_and_report = lambda coro, loop, what: _drain(coro)
    return lane


def _event(sym: str, *, side: str, px=None, tif="DAY", reason="venue refused"):
    return SimpleNamespace(
        instrument_id=SimpleNamespace(symbol=sym),
        client_order_id=f"SMHGLD-903-{sym}",
        order_side=SimpleNamespace(name=side),
        last_px=px,
        time_in_force=SimpleNamespace(name=tif),
        reason=reason,
    )


def test_a_fill_syncs_the_claim_to_the_ACTUAL_cache_quantity():
    lane = _lane(qty=129)

    lane.on_order_filled(_event("GLD", side="BUY", px=310.25))

    assert lane._runner.synced == [("GLD", 129, 310.25)]
    assert "GLD" in lane._held


def test_a_rejected_entry_syncs_zero_so_a_submit_time_claim_is_dropped():
    lane = _lane(qty=0, pending={"SMH": "enter"})

    lane.on_order_rejected(_event("SMH", side="BUY", px=None))

    assert lane._pending == {}
    assert lane._runner.synced == [("SMH", 0, None)]


def test_a_trim_fill_keeps_the_leg_held_and_syncs_the_remaining_quantity():
    lane = _lane(qty=83, held={"SMH"}, pending={"SMH": "exit"})

    lane.on_order_filled(_event("SMH", side="SELL", px=568.50))

    assert "SMH" in lane._held
    assert lane._runner.synced == [("SMH", 83, 568.50)]
