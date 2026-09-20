"""The PURE decision layer for the SMHGLD sleeve. No Nautilus, no database, no broker, no clock.

WHY PURE. The same function serves the backtest and the live path. Two implementations of one
decision drift, and the drift is invisible until a trade-for-trade comparison is run.

THE RULE, in full: hold both legs at their configured weights. Trade only when the risk leg's share
of the sleeve has drifted further than `drift_band` from its target. There is no trend gate, no
tilt and no regime — see `config.py` for the eleven mechanisms that were tested and discarded.

!! FEATURES ARE STILL SHIFTED even though nothing is being predicted. The prior close is what a
!! decision taken at this session's open may use, and the drift computed from it is the drift the
!! book actually carries into the day. The switching lane this replaces had a weekly signal built
!! with `resample().last()` and applied across its own week, which meant Monday held the position
!! implied by Friday; that single mistake turned a Sharpe of 0.78 into 2.57. Nothing here needs a
!! lookahead to work, which is exactly why one would go unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from kumo_strategies.strategies.smhgld_sleeve.config import SmhGldSleeveConfig


@dataclass(frozen=True)
class Decision:
    """WHAT the strategy wants, never HOW it is executed.

    Same vocabulary as every other lane. A strategy that invents its own decision schema is
    invisible to cross-strategy monitoring.
    """

    hold: tuple[str, ...] = ()
    enter: tuple[str, ...] = ()
    exit: tuple[str, ...] = ()
    #: Target fraction of the sleeve per leg. Unlike the ranking lanes, this is the decision itself
    #: rather than a sizing detail attached to one: `enter` and `exit` cannot express "same names,
    #: different proportions", which is the only trade this lane ever makes.
    weights: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    #: Three states, never two. `unknown` is not `on_target`, because a drift we could not compute
    #: is not a drift we measured and found acceptable.
    regime: str = "unknown"
    reasons: tuple[str, ...] = ()


def rebalance_dates(dates, period: str = "D") -> list[pd.Timestamp]:
    """The cadence rule, ONCE. Adapters and gateways call this rather than re-deriving it.

    FIRST session of each bucket, not the last. Deliberate, and it is the half that keeps the lane
    honest: acting on the first session of a week using features through the prior close is
    computable live. Taking the bucket's LAST value and applying it across the bucket is the
    look-ahead described in this module's header.
    """
    frame = pd.DataFrame({"date": pd.to_datetime(pd.Index(dates).unique())}).sort_values("date")
    if period.upper() == "D":
        return frame["date"].tolist()
    return (frame.assign(bucket=lambda d: d["date"].dt.to_period(period))
            .drop_duplicates("bucket", keep="first")["date"].tolist())


def build_feature_panel(bars: pd.DataFrame, cfg: SmhGldSleeveConfig) -> pd.DataFrame:
    """Prior-close price per leg. That is the entire feature set."""
    if cfg.price_field not in bars.columns:
        raise ValueError(f"bars missing price field {cfg.price_field!r}")
    missing = set(cfg.universe) - set(bars["ticker"].unique())
    if missing:
        raise ValueError(
            f"bars are missing {sorted(missing)}, which this lane names in its config. A sleeve "
            f"with one leg absent is not a degraded strategy, it is a concentrated bet on the "
            f"other one")

    panel = bars[bars["ticker"].isin(cfg.universe)].sort_values(["ticker", "date"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    grouped = panel.groupby("ticker", sort=False)

    panel["asof_close"] = grouped[cfg.price_field].shift(1)
    panel["sessions_seen"] = grouped.cumcount()
    panel["eligible"] = (
        (panel["sessions_seen"] >= cfg.warmup_sessions) & panel["asof_close"].notna())
    return panel


def current_weights(shares: dict[str, float], prices: dict[str, float]) -> dict[str, float]:
    """Fraction of the sleeve's own market value in each leg.

    OF THE SLEEVE, never of the account. A lane that measures its drift against the whole book
    rebalances itself every time an unrelated strategy opens a position.
    """
    values = {sym: shares.get(sym, 0.0) * prices[sym] for sym in prices}
    total = sum(values.values())
    if total <= 0.0:
        return {}
    return {sym: value / total for sym, value in values.items()}


def _refuse_a_frame_that_is_not_one_day(day: pd.DataFrame) -> None:
    """`decide` takes ONE SESSION. The parameter has been called `day` from the first commit and
    nothing made it true (#205).

    TWO WAYS TO VIOLATE IT, AND ONLY ONE OF THEM IS LOUD:

      NOT FEATURIZED    `KeyError: 'eligible'`. This killed every SMHGLD-007 rung on 2026-09-11 —
                        the caller passed `panel()` straight through, and `eligible`/`asof_close`
                        exist only after `build_feature_panel`.

      NOT DAY-SLICED    SILENT, and worse. A featurized frame holding the whole window HAS
                        `eligible`, so any presence check passes — and `.iloc[0]` below then takes
                        the OLDEST session in the window and returns a plausible trade at month-old
                        prices, with nothing anywhere saying so.

    A PRESENCE CHECK ON `eligible` CATCHES ONLY THE FIRST. That is the point of this function: the
    obvious guard repairs the loud half and leaves the half that trades. QC345-003 died of the same
    defect on its own first live rebalance (2026-08-20 13:35Z, post-mortem at
    `qc345_rotation.py:585-598`), and it survives review because BOTH FRAMES ARE CALLED `panel` and
    the featurization is a side effect of constructing something else.

    An EMPTY frame is allowed through: a lane with no rows for its session is a legitimate state the
    caller reports as a hold, and `decide` already answers it by holding.
    """
    if day is None:
        raise ValueError(
            "decide() was given None rather than a day-sliced feature frame. A missing frame is "
            "not an empty one: the caller has nothing to decide on and must hold, not call this.")
    missing = [c for c in ("eligible", "asof_close") if c not in day.columns]
    if missing:
        raise ValueError(
            f"decide() needs {missing} and the frame does not carry {'them' if len(missing) > 1 else 'it'}. "
            f"Those columns are produced by `build_feature_panel(bars, cfg)` — this looks like raw "
            f"`panel()` output passed straight through. Columns present: {sorted(day.columns)}.")
    if not day.empty:
        dates = pd.Series(day["date"]).dropna().unique()
        if len(dates) > 1:
            span = sorted(pd.Timestamp(d) for d in dates)
            raise ValueError(
                f"decide() takes ONE session and this frame holds {len(dates)}, "
                f"{span[0].date()} to {span[-1].date()}. It was featurized and NOT day-sliced, so "
                f"the `.iloc[0]` below would silently decide on {span[0].date()} — the OLDEST "
                f"session in the window, at prices that are {(span[-1] - span[0]).days} days stale. "
                f"Slice with `frame.loc[frame['date'] == session]` before calling. Do NOT slice on "
                f"`date.max()`: on a stale feed that decides on whatever row is newest, equally "
                f"silently, where `== session` yields an empty frame and the caller holds.")


def decide(day: pd.DataFrame, cfg: SmhGldSleeveConfig,
           held: set[str], shares: dict[str, float] | None = None) -> Decision:
    """One session's decision. `held` and `shares` are what THIS STRATEGY owns.

    Passing the account's positions here is how TECHIVOL-005 proposed exiting eight names belonging
    to two other strategies on its first live session.
    """
    _refuse_a_frame_that_is_not_one_day(day)
    rows = {sym: day.loc[day["ticker"] == sym] for sym in cfg.universe}
    if any(frame.empty or not bool(frame["eligible"].iloc[0]) for frame in rows.values()):
        # UNKNOWN HOLDS. Without a price for both legs the drift is unmeasurable, and rebalancing
        # on an unmeasurable drift would make every data gap a trade.
        return Decision(hold=tuple(sorted(held)), weights=cfg.target_weights, regime="unknown",
                        reasons=("one or both legs have no usable prior close; "
                                 "drift is not computable and the sleeve holds",))

    prices = {sym: float(frame["asof_close"].iloc[0]) for sym, frame in rows.items()}
    target = cfg.target_weights

    if not held:
        return Decision(
            hold=(), enter=tuple(sorted(cfg.universe)), weights=target, scores=prices,
            regime="opening",
            reasons=(f"sleeve is empty; opening at "
                     + ", ".join(f"{sym} {weight:.0%}" for sym, weight in sorted(target.items())),))

    actual = current_weights(shares or {}, prices)
    if not actual:
        return Decision(hold=tuple(sorted(held)), weights=target, regime="unknown",
                        reasons=("held legs have no market value; drift is not computable",))

    drift = {sym: actual.get(sym, 0.0) - target[sym] for sym in target}
    worst = max(drift, key=lambda sym: abs(drift[sym]))
    missing = tuple(sorted(set(cfg.universe) - held))

    if abs(drift[worst]) <= cfg.drift_band and not missing:
        return Decision(
            hold=tuple(sorted(held)), weights=target, scores=actual, regime="on_target",
            reasons=(f"{worst} at {actual.get(worst, 0.0):.1%} against a target of "
                     f"{target[worst]:.0%}, inside the {cfg.drift_band:.0%} band",))

    return Decision(
        # Both legs are wanted afterwards, so nothing is EXITED — the trade is a resize. Reporting
        # a resize as exit-then-enter would make the journal show a lane that flattens and reopens
        # its whole book every time it drifts a few points.
        hold=tuple(sorted(held & set(cfg.universe))),
        enter=missing,
        weights=target, scores=actual, regime="rebalance",
        reasons=(f"{worst} at {actual.get(worst, 0.0):.1%} against a target of "
                 f"{target[worst]:.0%}, outside the {cfg.drift_band:.0%} band",)
        + ((f"legs absent from the sleeve: {', '.join(missing)}",) if missing else ()),
    )


@dataclass(frozen=True)
class Order:
    """One side's worth of work, in shares. `delta > 0` buys, `delta < 0` sells.

    SHARES, not notional, because the thing that reaches the broker is a quantity and every
    rounding decision belongs on this side of the boundary where it can be tested.
    """

    symbol: str
    delta: float
    target_qty: float
    held_qty: float
    reason: str

    @property
    def post_trade_qty(self) -> float:
        """What the book holds once this order fills. THE NUMBER THE CLAIM IS SET TO.

        `held_qty + delta`, NEVER `target_qty`. The delta is rounded to a whole lot and the target
        is not, so they differ by up to one lot on every resize — and a claim set to the unrounded
        target is wrong by that much from the moment it is written, permanently, with nothing that
        would later disagree.

        Claims are synced ABSOLUTELY (`sync_claim_to`), not by delta, and that is deliberate:
        re-stating the truth is idempotent, so a missed call, a duplicate call or a reordered pair
        all converge on the book. A delta API handed this lane's own sequence — 100, 93, 88, 88
        with no trade, 0 — applies the 88 twice and takes the claim to zero on a live position.
        """
        return self.held_qty + self.delta


def order_plan(decision: Decision, shares: dict[str, float], prices: dict[str, float],
               equity: float, *, lot: float = 1.0) -> tuple[Order, ...]:
    """Turn target weights into the buys and sells that reach them.

    2026-09-04: *"There is a sizing. The sizing needs to be executed. That can mean buys or
    sells of certain quantities of a symbol. Simple."* This is that, for two legs:

        target_qty = weight * equity / price
        delta      = target_qty - held_qty

    An entry is a delta from zero and an exit is a target of zero, so the runner needs no special
    cases for either. Orders are returned SELLS FIRST: the sells fund the buys, and submitting the
    buy first on a fully-invested sleeve asks for margin the lane does not have.

    `lot` rounds toward the current holding, never away, so rounding can only ever leave the sleeve
    closer to where it already is than to an overshoot it would have to undo next session.
    """
    if decision.regime not in {"opening", "rebalance"}:
        return ()

    orders = []
    for symbol, weight in sorted(decision.weights.items()):
        price = prices[symbol]
        if price <= 0:
            raise ValueError(f"{symbol} priced at {price}; a target computed from it is not a size")

        held = shares.get(symbol, 0.0)
        target = weight * equity / price
        delta = target - held
        # Round the DELTA toward zero, so a rounding remainder leaves the position where it is
        # rather than pushing it past target and inviting the opposite trade next session.
        delta = (abs(delta) // lot) * lot * (1.0 if delta > 0 else -1.0)
        if delta == 0.0:
            continue

        orders.append(Order(
            symbol=symbol, delta=delta, target_qty=target, held_qty=held,
            reason=f"{'buy' if delta > 0 else 'sell'} {abs(delta):.0f} to reach "
                   f"{weight:.2%} of {equity:,.0f}"))

    return tuple(sorted(orders, key=lambda o: o.delta))
