"""Configuration for CRSISHORT — the overbought high-volatility short book (#123).

NAME CARRIES NO NUMBER (AGENTS.md). Cockpit allocates the `order_id_tag` via `assign_tag("CRSISHORT")`.

Every field the lab froze is here, including the ones that look like execution detail. The entry
LIMIT is not execution detail: #123 measures +79.7% on 228 trades with the limit and +99.4% on 384
without it, identical return/DD, worse tail, and the market version LOSES above 100bps of cost. A
runner that took the limit from a call site could measure one strategy and deploy the other.
"""

from __future__ import annotations

from dataclasses import dataclass
@dataclass(frozen=True)
class CrsiShortConfig:
    """FROZEN, so a runtime layer cannot quietly mutate what the backtest validated."""

    # --- universe (gates ENTRIES ONLY; see engine.decide) ---------------------------------
    min_dollar_volume: float = 200_000_000.0
    """Previous session's close x volume. #123: the edge is in LIQUID names — a $500M floor gives
    +56.8% on 126 trades at -5% DD, and a $50M floor degrades to +34.3% at -32%."""

    price_floor: float = 5.0

    manage_min_dollar_volume: float = 1_000_000.0
    """The floor below which a name stops being PRICEABLE, as opposed to enterable.

    Two floors, and they do different jobs. `min_dollar_volume` ($200M) gates entries and is held
    through — a position whose liquidity dips is managed to its exit. This one is the lab's signal
    table filter: below it (or with a NaN indicator) the name leaves the table entirely, and a held
    position is closed at its last mark and booked `nodata`. That is 4% of #123's trades, so it is
    a rule with a measured cost, not a data-quality footnote."""

    # --- filter ---------------------------------------------------------------------------
    min_annual_vol: float | None = 1.0
    """Annualised 100-session log-return vol. 1.0 is 100%/yr. `None` disables the filter, which is
    one of the four mutations #123's acceptance requires to MOVE the result."""

    vol_window: int = 100

    # --- entry ----------------------------------------------------------------------------
    crsi_entry: float = 90.0
    crsi_rsi_period: int = 3
    crsi_streak_period: int = 2
    crsi_rank_period: int = 100

    entry_limit_pct: float = 0.03
    """Sell-short limit this far ABOVE the signal close, resting one session. Monotone in mean/trade
    across +0/1/2/3/5% (4.22 -> 6.22%); +3% is the original author's number and the Sharpe peak."""

    # --- exits ----------------------------------------------------------------------------
    reversal_exit: bool = True
    """Cover after a close above the PRIOR session's high, at the next open."""

    weakness_exit: bool = False
    """Cover after a close below the prior session's LOW. OFF in the frozen spec: it covers a short
    that is winning, and the lab measures it as the weakest of the four exit sets."""

    give_back_frac: float | None = 1.0
    """The FLAT exit, as a give-back trail (#110). 1.0 = never return to entry once in profit.
    Worth 8 points of return here and takes the drawdown from -13.4% to -8.1%."""

    # --- portfolio ------------------------------------------------------------------------
    n_slots: int = 20
    """20 slots, so a slot is 5% of equity. Sizing is `equity / n_slots * size_multiple`."""

    size_multiple: float = 1.0
    """Multiple of an equal slot. 1.0 is the measured book.

    Expressed as a MULTIPLE rather than as a second weight field, so the slot count and the slot
    size cannot disagree: `n_slots=20, slot_weight=0.10` typechecks and silently doubles the book.
    #123 asks for this knob explicitly — a -632% position return is -31% of equity at a full slot
    and -16% at `size_multiple=0.5` — and says the paper gate, not the observed -8.1% drawdown,
    should set it."""

    hold_through: bool = True
    """Whether a held name that drops below the ENTRY liquidity floor is managed to its exit (True)
    or closed (False). #123 freezes True; the lab measures both. False is not a stricter variant,
    it is the accident the earlier code had: it closes winners for a reason unrelated to the trade."""

    max_borrow_fee_annual: float | None = 1.0
    """Pre-trade ceiling on the locate fee, 1.0 = 100%/yr. Break-even borrow is 216%/yr, so this
    gate is not about the cost: the >100% bucket goes gross +2.55% to net +0.56%, i.e. an expensive
    borrow marks a name whose edge is already gone. Enforced where locate data exists; `None` off."""

    max_stale_days: int = 4
    """Maximum calendar gap from the newest completed daily bar to the session being decided.

    The pre-open decision runs before today's daily bar exists, so the adapter adds a placeholder row
    for that session and shifts features from the newest completed bar. That makes a missing session
    row normal, but it also removes the old empty-slice signal for a stale feed; this guard restores a
    named refusal for real outages. Four days admits a normal Friday-to-Monday gap."""

    @property
    def warmup_sessions(self) -> int:
        """Sessions of history before this strategy may signal at all.

        DERIVED, never stored. A stored warmup that disagreed with the windows would let the first
        sessions signal on a half-formed statistic. `max(rank, vol) + 2` is the lab's own guard
        (`len(df) < max(RANK_P, VOL_P) + 2` in `s411_build_signals.py`): the rank period counts
        RETURNS, which costs one session, and this layer shifts everything one further so the
        decision for session `d` is computable from `d-1`.
        """
        return max(self.vol_window, self.crsi_rank_period) + 2

    def tag(self) -> str:
        """Artifact identity. Every swept field belongs here or two runs overwrite each other."""
        vol = "off" if self.min_annual_vol is None else f"{self.min_annual_vol:g}"
        gb = "off" if self.give_back_frac is None else f"{self.give_back_frac:g}"
        return (f"crsi-{self.crsi_entry:g}__vol-{vol}__dv-{self.min_dollar_volume:g}"
                f"__lim-{self.entry_limit_pct:g}__slots-{self.n_slots}__x-{self.size_multiple:g}"
                f"__gb-{gb}__rev-{int(self.reversal_exit)}__weak-{int(self.weakness_exit)}"
                f"__hold-{int(self.hold_through)}")

    def __post_init__(self) -> None:
        from kumo_strategies.strategies._validate import check_field_types

        check_field_types(self)
        if self.give_back_frac is not None and not 0 < self.give_back_frac <= 1.0:
            raise ValueError(
                f"give_back_frac must be in (0, 1], got {self.give_back_frac!r}. Use None to "
                "disable the flat exit rather than 0, which reads as 'give back nothing' and "
                "fires on every position immediately.")
        if self.entry_limit_pct < 0:
            raise ValueError(
                f"entry_limit_pct must be >= 0, got {self.entry_limit_pct!r}. A NEGATIVE limit "
                "would rest BELOW the signal close and fill on weakness — the opposite of the "
                "rule, and it would still produce trades and a plausible curve.")
        if self.n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {self.n_slots!r}")
        if self.size_multiple <= 0:
            raise ValueError(f"size_multiple must be > 0, got {self.size_multiple!r}")

    def slot_notional(self, equity: float) -> float:
        """What one entry is sized to. ONE definition, so no runner grows a second one.

        `equity / n_slots` and not `equity * weight`: the two are the same number only when the
        weight is exactly 1/n, and `int(notional / price)` turns a last-bit difference into a
        one-share difference in the book — which is enough to make two equity curves diverge and
        cost a day finding out why."""
        return equity / self.n_slots * self.size_multiple
