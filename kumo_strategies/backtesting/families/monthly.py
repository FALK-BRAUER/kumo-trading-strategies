"""The MONTHLY family for the one runner (`engine.SessionEngine`): QC27 tech inverse-vol and
QC345 rotation — a rebalance calendar, target-weight sizing, the QC27 portfolio stop and cash-proxy
sweep, the QC345 ATR trail and delisting marks.

Faithful decompositions of `runner_qc27_verified.run` / `runner_qc345_verified.run` (moved whole
into `monthly.py` at #270 step 3, split at the engine's hooks here): the same statements in the
same order. The gate is each strategy's recorded number reproduced to the cent (QC27 193.4707 %,
QC345 176.7125 %, 2025 → 2026-08-10, the lab's daily bars).

STILL CALL-SITE KNOBS, named as debt: `fill_price_col`/`fill_hour`/`fill_minute` (the slot-price
arm of research/residual-gate), `rebalance_period`, `rebalance_holds`, `delisting_mode`. The unified
config surface (docs/runner-architecture.md §3) takes them as fields; until then they stay arguments
so every recorded research result reproduces.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pandas as pd

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.engine import Order, SessionEngine
from kumo_strategies.backtesting.sim_venue import Book, LabelDefs, SimVenue
from kumo_strategies.strategies import qc27_tech_inverse_vol as QC27
from kumo_strategies.strategies import qc345_rotation as QC345
from kumo_strategies.strategies.market_view import (
    MarketSignal,
    MarketViewConfig,
    market_state,
    target_weights_under,
)
from kumo_strategies.strategies.momentum_rotation.engine import trailing_atr
from kumo_strategies.strategies.momentum_rotation.exits import TrailState, evaluate_exits, needs_atr
from kumo_strategies.strategies.qc27_tech_inverse_vol import QC27TechInverseVolConfig
from kumo_strategies.strategies.qc345_rotation import QC345RotationConfig
from kumo_strategies.strategies.qc345_rotation.source import QC345ComputedSource

# ==== QC27 ====

class QC27Monthly:
    """QC27 tech inverse-vol on the monthly calendar: target weights at each rebalance, a
    portfolio-fraction stop checked on the close and executed at the next fill instant, a cash-proxy
    sweep of idle cash every session."""

    def __init__(self, bars: pd.DataFrame, sectors: pd.DataFrame, *, cfg: QC27TechInverseVolConfig,
                 cost_model: CostModel, starting_cash: float, n_trials: int,
                 benchmarks: dict[str, pd.Series] | None, rebalance_period: str | None,
                 fill_price_col: str, fill_hour: int, fill_minute: int, rebalance_holds: bool,
                 market_view: MarketViewConfig | None) -> None:
        if fill_price_col not in bars.columns:
            raise ValueError(f"bars has no fill price column {fill_price_col!r}; "
                             f"available: {sorted(bars.columns)}")
        px = bars.copy()
        px["date"] = pd.to_datetime(px["date"])
        px = px.sort_values(["ticker", "date"]).reset_index(drop=True)
        tech_bars, snapshot_diag = QC27.filter_tech_universe(px, sectors, cfg)
        scored, self.selection_diag = QC27.build_feature_panel(tech_bars, cfg, snapshot_diag)
        self.tech_lookup = {d: frame for d, frame in scored.groupby("date", sort=False)}
        # THE LANE'S OWN MARKET, from the same tech universe the lane ranks. The CONFIG is the
        # declaration; the argument is an override for sweeps.
        self.mv = market_view if market_view is not None else cfg.market_view
        self.market_panel = (
            tech_bars.pivot_table(index="date", columns="ticker", values="close").sort_index()
            if self.mv.signal is not MarketSignal.NONE else None)
        self.market_states: dict[object, object] = {}
        self.all_lookup = {d: frame.set_index("ticker") for d, frame in px.groupby("date", sort=False)}
        # None means "whatever the config says": the DEFAULT cannot drift from production.
        self.rebalances = set(QC27.rebalance_dates(px["date"], rebalance_period or cfg.rebalance_period))
        self.sessions = sorted(self.all_lookup)

        self.cfg = cfg
        self.col, self.at = fill_price_col, (fill_hour, fill_minute)
        self.rebalance_holds = rebalance_holds
        self.starting_cash, self.n_trials, self.benchmarks = starting_cash, n_trials, benchmarks
        self.book = Book(starting_cash)
        # `defs` here is the per-symbol MIC the fill row records; QC27 never built instruments.
        self.venue = SimVenue(cost_model, LabelDefs("SIM"))
        self.engine = SessionEngine(self.book, self.venue)
        self.basis: dict[str, float] = {}
        self.last_close: dict[str, float] = {}
        self.pending_stop_exits: set[str] = set()
        self.cash_shortfall_entry_skips = 0
        self.missing_position_liquidations = 0
        self.stop_trigger_count = 0
        self.stop_exit_count = 0
        self.gld_sweep_buys = 0
        self.untradeable_at_fill_time = 0
        self._decisions: list[dict[str, object]] = []
        self._target_qty: dict[str, int] = {}

    def _ts(self, d):
        return pd.Timestamp(d) + pd.Timedelta(hours=self.at[0], minutes=self.at[1])

    def skip(self, d) -> bool:
        return False

    def session_open(self, d) -> list[Order]:
        held, day_prices, ts, col = self.book.held, self.all_lookup[d], self._ts(d), self.col
        out: list[Order] = []
        for symbol in sorted(list(held)):
            if symbol in day_prices.index:
                continue
            qty = held.pop(symbol)
            px_last = self.last_close.get(symbol, 0.0)
            if qty > 0 and px_last > 0:
                out.append(Order(symbol, "SELL", qty, px_last, ts, "liquidate", label="MISSING"))
                self.missing_position_liquidations += 1
            self.basis.pop(symbol, None)
            self.pending_stop_exits.discard(symbol)
        for symbol in sorted(list(self.pending_stop_exits)):
            if symbol not in held or symbol not in day_prices.index:
                self.pending_stop_exits.discard(symbol)
                continue
            if not pd.notna(day_prices.loc[symbol, col]):
                self.untradeable_at_fill_time += 1
                continue   # stays pending; sold at the next session that has a price
            qty = held.pop(symbol)
            out.append(Order(symbol, "SELL", qty, float(day_prices.loc[symbol, col]), ts, "set", at=self.at))
            self.basis.pop(symbol, None)
            self.pending_stop_exits.discard(symbol)
            self.stop_exit_count += 1
        return out

    def slots(self, d) -> list:
        if d not in self.rebalances:
            return []
        feature_day = self.tech_lookup.get(d)
        if feature_day is None or feature_day.empty:
            return []
        self._feature_day = feature_day
        return [("rebalance", None)]

    def before_decision(self, d, slot) -> None:
        return None

    def decide(self, d, slot) -> bool:
        cfg, held, day_prices, col = self.cfg, self.book.held, self.all_lookup[d], self.col
        tech_held = {sym for sym in held if sym != cfg.cash_proxy_symbol}
        dec = QC27.decide(self._feature_day, cfg, tech_held)
        current_equity = self.book.cash
        for symbol, qty in held.items():
            # A held name with no print at this hour is still WORTH something — it just cannot be
            # traded now; a NaN here would size the whole book to zero.
            px_open = self.last_close.get(symbol, 0.0)
            if symbol in day_prices.index:
                _p = day_prices.loc[symbol, col]
                if pd.notna(_p):
                    px_open = float(_p)
            current_equity += qty * px_open
        # RISK-OFF FLATTENS AND BLOCKS, in one place (rule 3 through the shared function, #170);
        # UNKNOWN deliberately does NOT flatten (#873).
        view = market_state(self.market_panel, self.mv, asof=d) if self.market_panel is not None else None
        if view is not None:
            self.market_states[d] = view
        target_weights = target_weights_under(view, dec.weights, held)
        if dec.cash_proxy_weight > 0 and cfg.cash_proxy_symbol in day_prices.index:
            target_weights[cfg.cash_proxy_symbol] = dec.cash_proxy_weight
        target_qty: dict[str, int] = {}
        for symbol, weight in target_weights.items():
            if symbol not in day_prices.index:
                continue
            if not pd.notna(day_prices.loc[symbol, col]):
                self.untradeable_at_fill_time += 1
                continue
            open_px = float(day_prices.loc[symbol, col])
            if open_px <= 0:
                continue
            # `rebalance_holds=False` sizes a name ONCE, at entry — what the live runner does (#68).
            target_qty[symbol] = (
                held[symbol] if (not self.rebalance_holds and symbol in held)
                else int((current_equity * weight) // open_px))
        self._target_qty, self._dec, self._view = target_qty, dec, view
        return True

    def exit_orders(self, d, slot) -> list[Order]:
        held, day_prices, ts, col = self.book.held, self.all_lookup[d], self._ts(d), self.col
        out: list[Order] = []
        for symbol in sorted(list(held)):
            want = self._target_qty.get(symbol, 0)
            have = held.get(symbol, 0)
            if have <= want or symbol not in day_prices.index:
                continue
            if not pd.notna(day_prices.loc[symbol, col]):
                self.untradeable_at_fill_time += 1
                continue
            qty = have - want
            out.append(Order(symbol, "SELL", qty, float(day_prices.loc[symbol, col]), ts, "set", at=self.at))
            held[symbol] = want
            if want == 0:
                held.pop(symbol, None)
                self.basis.pop(symbol, None)
        return out

    def entry_orders(self, d, slot) -> list[Order]:
        book, venue, held, day_prices, ts, col = self.book, self.venue, self.book.held, self.all_lookup[d], self._ts(d), self.col
        for symbol, want in sorted(self._target_qty.items()):
            have = held.get(symbol, 0)
            if want <= have:
                continue
            qty = want - have
            price = float(day_prices.loc[symbol, col])
            if not venue.budget_allows(book, symbol, qty, price, ts, at=self.at):
                self.cash_shortfall_entry_skips += 1
                continue
            self.engine.execute([Order(symbol, "BUY", qty, price, ts, "set", at=self.at)])
            prev_qty = held.get(symbol, 0)
            prev_basis = self.basis.get(symbol, 0.0)
            new_qty = prev_qty + qty
            self.basis[symbol] = ((prev_basis * prev_qty) + (price * qty)) / new_qty
            held[symbol] = new_qty
        dec, view = self._dec, self._view
        self._decisions.append({
            "date": d, "hold": list(dec.hold), "scores": dec.scores, "weights": dec.weights,
            "cash_proxy_weight": dec.cash_proxy_weight,
            "market_state": None if view is None else view.state,
            "market_reasons": None if view is None else "; ".join(view.reasons),
        })
        return []

    def session_close(self, d) -> list[Order]:
        """The cash-proxy sweep: idle cash into the proxy every session, unless risk-off."""
        book, venue, held, day_prices, ts, col, cfg = self.book, self.venue, self.book.held, self.all_lookup[d], self._ts(d), self.col, self.cfg
        _risk_off = getattr(self.market_states.get(d), "liquidates", False)
        if book.cash > 0 and not _risk_off and cfg.cash_proxy_symbol in day_prices.index:
            _g = day_prices.loc[cfg.cash_proxy_symbol, col]
            gld_open = float(_g) if pd.notna(_g) else 0.0
            qty = int(book.cash // gld_open) if gld_open > 0 else 0
            if qty > 0 and venue.budget_allows(book, cfg.cash_proxy_symbol, qty, gld_open, ts, at=self.at):
                self.engine.execute([Order(cfg.cash_proxy_symbol, "BUY", qty, gld_open, ts, "set", at=self.at)])
                prev_qty = held.get(cfg.cash_proxy_symbol, 0)
                prev_basis = self.basis.get(cfg.cash_proxy_symbol, 0.0)
                new_qty = prev_qty + qty
                self.basis[cfg.cash_proxy_symbol] = ((prev_basis * prev_qty) + (gld_open * qty)) / new_qty
                held[cfg.cash_proxy_symbol] = new_qty
                self.gld_sweep_buys += 1
        return []

    def marks(self, d) -> dict[str, float]:
        day_prices, out = self.all_lookup[d], {}
        for symbol in self.book.held:
            if symbol in day_prices.index:
                px = float(day_prices.loc[symbol, "close"])
                self.last_close[symbol] = px
            else:
                px = self.last_close.get(symbol, 0.0)
            out[symbol] = px
        return out

    def curve_row(self, d, value: float) -> bool:
        self._equity = value
        return True

    def after_close(self, d) -> None:
        """The portfolio-fraction stop, checked on the close; the exit fills at the next instant."""
        cfg, held, day_prices, equity = self.cfg, self.book.held, self.all_lookup[d], self._equity
        threshold = cfg.stop_loss_portfolio_frac * equity if equity > 0 else 0.0
        for symbol, qty in held.items():
            if symbol == cfg.cash_proxy_symbol or symbol not in day_prices.index:
                continue
            close_px = float(day_prices.loc[symbol, "close"])
            drawdown_value = max(0.0, (self.basis.get(symbol, close_px) - close_px) * qty)
            if drawdown_value > threshold:
                self.pending_stop_exits.add(symbol)
                self.stop_trigger_count += 1

    def decisions(self) -> list[dict]:
        return self._decisions

    def diagnostics(self) -> dict:
        sd = self.selection_diag
        return {
            "sector_snapshot_symbols": sd.sector_snapshot_symbols,
            "tech_snapshot_symbols": sd.tech_snapshot_symbols,
            "eligible_name_days": sd.eligible_name_days,
            "positive_name_days": sd.positive_name_days,
            "cash_shortfall_entry_skips": self.cash_shortfall_entry_skips,
            "missing_position_liquidations": self.missing_position_liquidations,
            "stop_trigger_count": self.stop_trigger_count,
            "stop_exit_count": self.stop_exit_count,
            "gld_sweep_buys": self.gld_sweep_buys,
            "untradeable_at_fill_time": self.untradeable_at_fill_time,
            "fill_price_col": self.col,
            "fill_time": f"{self.at[0]:02d}:{self.at[1]:02d}",
            "cash_proxy_symbol": self.cfg.cash_proxy_symbol,
            "universe_caveat": "technology universe comes from a current snapshot sectors.parquet, not point-in-time QC sector membership",
        }


# ==== QC345 ====

_EXCHANGE_TO_MIC = {"NASDAQ": "XNAS", "NYSE": "XNYS", "ARCA": "ARCX", "AMEX": "XASE", "BATS": "BATS"}


def _load_instruments(path: str | Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text())


def _fallback_metadata(symbol: str, assets_row: pd.Series | None) -> dict[str, str]:
    exchange = str(assets_row.get("exchange", "") if assets_row is not None else "")
    return {"symbol": symbol, "mic": _EXCHANGE_TO_MIC.get(exchange.upper(), "XNAS"),
            "price_increment": "0.01", "exchange": exchange, "fallback": True}


def _symbol_metadata(symbols: set[str], assets: pd.DataFrame | None, instruments_path: str | Path,
                     ) -> tuple[dict[str, dict], dict[str, int]]:
    instruments = _load_instruments(instruments_path)
    assets_index = (assets.drop_duplicates("symbol").set_index("symbol")
                    if assets is not None and "symbol" in assets.columns
                    else pd.DataFrame().set_index(pd.Index([], name="symbol")))
    meta: dict[str, dict] = {}
    missing_assets = fallback_count = missing_instruments = 0
    for symbol in sorted(symbols):
        if symbol in instruments:
            meta[symbol] = instruments[symbol]
            continue
        missing_instruments += 1
        if symbol in assets_index.index:
            meta[symbol] = _fallback_metadata(symbol, assets_index.loc[symbol])
            fallback_count += 1
            continue
        missing_assets += 1
    return meta, {"symbols_missing_instrument_metadata": missing_instruments,
                  "symbols_using_fallback_metadata": fallback_count,
                  "symbols_dropped_missing_assets_metadata": missing_assets}


def _mark_position(row: pd.Series | None, qty: int, terminal_bucket: str, delisting_mode: str,
                   last_known_close: float) -> float:
    if row is not None:
        return float(row["close"]) * qty
    if terminal_bucket == "bankruptcy":
        return 0.0
    if delisting_mode == "lastpx":
        return last_known_close * qty
    if terminal_bucket == "unclassified" and delisting_mode == "wipeout_unclassified":
        return 0.0
    return last_known_close * qty


def _has_exit_rules(cfg: QC345RotationConfig) -> bool:
    return any(getattr(cfg.exits, f.name) is not None for f in fields(cfg.exits))


def _atr_for(px: pd.DataFrame, day: pd.Timestamp, cfg: QC345RotationConfig) -> dict[str, float] | None:
    """Per-symbol ATR as at `day`, or None when no configured rule needs it. Asks `needs_atr`
    rather than naming fields; `trailing_atr` is the ONE implementation, shared."""
    if not needs_atr(cfg.exits):
        return None
    missing = [c for c in ("high", "low") if c not in px.columns]
    if missing:
        raise ValueError(
            f"an ATR-scaled exit rule is configured but the panel has no {' and '.join(missing)} "
            f"column: ATR is true RANGE, so high and low are required. Supply them, or clear "
            f"{'/'.join(f for f in ('stop_loss_atr', 'take_profit_atr', 'give_back_min_peak_atr') if getattr(cfg.exits, f) is not None)}.")
    return trailing_atr(px[px["date"] <= day])


class QC345Monthly:
    """QC345 rotation on the monthly calendar: equal-notional targets, the ATR trail between
    rebalances, delisting marks by terminal bucket."""

    def __init__(self, bars: pd.DataFrame, assets: pd.DataFrame, *, cfg: QC345RotationConfig,
                 instruments_path: str | Path, cost_model: CostModel, starting_cash: float,
                 deployed: float, delisting_mode: str, n_trials: int,
                 benchmarks: dict[str, pd.Series] | None, fill_price_col: str, fill_hour: int,
                 fill_minute: int, market_view: MarketViewConfig | None) -> None:
        if fill_price_col not in bars.columns:
            raise ValueError(f"bars has no fill price column {fill_price_col!r}")
        px = bars.copy()
        px["date"] = pd.to_datetime(px["date"])
        px = QC345.filter_asset_universe(px, assets, cfg).sort_values(["ticker", "date"]).reset_index(drop=True)
        metadata, self.metadata_diag = _symbol_metadata(set(px["ticker"]), assets, instruments_path)
        px = px[px["ticker"].isin(set(metadata))].copy()
        self.source = QC345ComputedSource(px, assets, cfg)
        px = self.source.filtered_bars
        self.px, scored = px, self.source.panel
        self.rebalances = set(QC345.rebalance_dates(px["date"]))
        self.terminal_bucket = QC345.terminal_buckets(px, assets)
        self.date_lookup = {d: frame.set_index("ticker") for d, frame in px.groupby("date", sort=False)}
        self.feature_lookup = {d: frame for d, frame in scored.groupby("date", sort=False)}
        self.sessions = sorted(self.date_lookup)
        self.last_close_by_symbol: dict[str, float] = {}
        self.cfg, self.col, self.at = cfg, fill_price_col, (fill_hour, fill_minute)
        self.deployed, self.delisting_mode = deployed, delisting_mode
        self.starting_cash, self.n_trials, self.benchmarks = starting_cash, n_trials, benchmarks
        self.book = Book(starting_cash)
        # the venue label per fill is the symbol's MIC from the asset metadata, as always recorded
        self.venue = SimVenue(cost_model, LabelDefs("XNAS", {s: str(m["mic"]) for s, m in metadata.items()}))
        self.engine = SessionEngine(self.book, self.venue)
        self.trail_state: dict[str, TrailState] = {}
        self.cash_shortfall_count = 0
        self._decisions: list[dict[str, object]] = []
        self.exits_enabled = _has_exit_rules(cfg)
        # INERT BY DEFAULT: `signal=NONE` yields RISK_ON, every recorded QC345 number unchanged.
        self.mv = market_view if market_view is not None else cfg.market_view
        self.market_panel = None
        if self.mv.signal is not MarketSignal.NONE:
            self.market_panel = (bars.pivot_table(index="date", columns="ticker", values="close").sort_index())
        self.market_states: dict[object, object] = {}

    def _ts(self, d):
        return pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=35)

    def _mark(self, symbol, qty, row):
        return _mark_position(row=row, qty=qty, terminal_bucket=self.terminal_bucket.get(symbol, "active"),
                              delisting_mode=self.delisting_mode,
                              last_known_close=self.last_close_by_symbol.get(symbol, 0.0))

    def skip(self, d) -> bool:
        return False

    def session_open(self, d) -> list[Order]:
        """Delistings written off at the last mark; then the ATR/give-back trail's forced exits at
        the fill instant."""
        held, day_prices, ts, col = self.book.held, self.date_lookup[d], self._ts(d), self.col
        out: list[Order] = []
        for symbol in sorted(list(held)):
            if symbol in day_prices.index:
                continue
            qty = held.pop(symbol)
            px_out = self._mark(symbol, 1, None)
            out.append(Order(symbol, "SELL", qty, px_out, ts, "liquidate", label="DELIST"))
            self.trail_state.pop(symbol, None)
        forced_exit_reasons: dict[str, str] = {}
        if self.exits_enabled and held:
            exit_prices = {symbol: self.last_close_by_symbol[symbol]
                           for symbol in held if symbol in self.last_close_by_symbol}
            plan = evaluate_exits(self.cfg.exits, exit_prices,
                                  {symbol: self.trail_state[symbol] for symbol in held if symbol in self.trail_state},
                                  atr=_atr_for(self.px, d, self.cfg))
            self.trail_state.update(plan.state)
            forced_exit_reasons = plan.exits
        self._forced_reasons = forced_exit_reasons
        self._forced = tuple(sorted(forced_exit_reasons))
        for symbol in self._forced:
            if symbol not in day_prices.index or symbol not in held:
                continue
            open_px = float(day_prices.loc[symbol, col])
            qty = held.pop(symbol)
            out.append(Order(symbol, "SELL", qty, open_px, ts, "set", at=self.at))
            self.trail_state.pop(symbol, None)
        return out

    def slots(self, d) -> list:
        forced = self._forced
        if d not in self.rebalances:
            if forced:
                self._decisions.append({"date": d, "rebalance": False, "hold": "", "enter": "",
                                        "exit": ",".join(forced), "forced_exit": ",".join(forced),
                                        "forced_exit_reason": self._forced_reasons, "scores": {}})
            return []
        feature_day = self.feature_lookup.get(d)
        if feature_day is None or feature_day.empty:
            if forced:
                self._decisions.append({"date": d, "rebalance": True, "hold": "", "enter": "",
                                        "exit": ",".join(forced), "forced_exit": ",".join(forced),
                                        "forced_exit_reason": self._forced_reasons, "scores": {}})
            return []
        self._feature_day = feature_day
        return [("rebalance", None)]

    def before_decision(self, d, slot) -> None:
        return None

    def decide(self, d, slot) -> bool:
        cfg, held, day_prices, col = self.cfg, self.book.held, self.date_lookup[d], self.col
        feature_day, forced = self._feature_day, self._forced
        universe = self.source.eligible(d)
        if forced:
            feature_day = feature_day.loc[~feature_day["ticker"].isin(forced)].copy()
            universe -= set(forced)
        dec = QC345.decide(feature_day, cfg, set(held), universe=universe)
        current_equity = self.book.cash + sum(
            (float(day_prices.loc[symbol, col]) * qty if symbol in day_prices.index
             else self._mark(symbol, qty, None))
            for symbol, qty in held.items())
        self._target_notional = current_equity * self.deployed / max(len(dec.hold), 1) if dec.hold else 0.0
        exits = tuple(sorted(set(dec.exit) | set(forced)))
        enters = tuple(sym for sym in dec.enter if sym not in self._forced_reasons)
        # RULE 3 THROUGH THE SHARED FUNCTION (#170), on entries and exits.
        view = market_state(self.market_panel, self.mv, asof=d) if self.market_panel is not None else None
        if view is not None:
            self.market_states[d] = view
            if view.liquidates:
                exits = tuple(sorted(set(exits) | set(held)))
                enters = ()
            elif view.blocks_entries:
                enters = ()
        self._dec, self._exits, self._enters = dec, exits, enters
        return True

    def exit_orders(self, d, slot) -> list[Order]:
        held, day_prices, ts, col = self.book.held, self.date_lookup[d], self._ts(d), self.col
        out: list[Order] = []
        for symbol in self._exits:
            if symbol not in day_prices.index or symbol not in held:
                continue
            open_px = float(day_prices.loc[symbol, col])
            qty = held.pop(symbol)
            out.append(Order(symbol, "SELL", qty, open_px, ts, "set", at=self.at))
            self.trail_state.pop(symbol, None)
        return out

    def entry_orders(self, d, slot) -> list[Order]:
        book, venue, held, day_prices, ts, col = self.book, self.venue, self.book.held, self.date_lookup[d], self._ts(d), self.col
        dec, target_notional = self._dec, self._target_notional
        for symbol in [s for s in dec.hold if s in held]:
            if symbol not in day_prices.index:
                continue
            open_px = float(day_prices.loc[symbol, col])
            target_qty = int(target_notional / open_px)
            delta = target_qty - held[symbol]
            if delta == 0:
                continue
            side = "BUY" if delta > 0 else "SELL"
            qty = abs(delta)
            if side == "BUY":
                if not venue.budget_allows(book, symbol, qty, open_px, ts, at=self.at):
                    self.cash_shortfall_count += 1
                    continue
                self.engine.execute([Order(symbol, "BUY", qty, open_px, ts, "set", at=self.at)])
                held[symbol] += qty
            else:
                self.engine.execute([Order(symbol, "SELL", qty, open_px, ts, "set", at=self.at)])
                held[symbol] -= qty
                if held[symbol] <= 0:
                    held.pop(symbol, None)
                    self.trail_state.pop(symbol, None)
        for symbol in self._enters:
            if symbol in held or symbol not in day_prices.index:
                continue
            open_px = float(day_prices.loc[symbol, col])
            qty = int(target_notional / open_px)
            if qty < 1:
                continue
            if not venue.budget_allows(book, symbol, qty, open_px, ts, at=self.at):
                self.cash_shortfall_count += 1
                continue
            self.engine.execute([Order(symbol, "BUY", qty, open_px, ts, "set", at=self.at)])
            held[symbol] = qty
            self.trail_state[symbol] = TrailState(entry_px=open_px, peak_px=open_px)
        self._decisions.append({"date": d, "rebalance": True, "hold": ",".join(dec.hold),
                                "enter": ",".join(self._enters), "exit": ",".join(self._exits),
                                "forced_exit": ",".join(self._forced),
                                "forced_exit_reason": self._forced_reasons, "scores": dec.scores})
        return []

    def session_close(self, d) -> list[Order]:
        return []

    def marks(self, d) -> dict[str, float]:
        """Per-share marks (the engine multiplies by the quantity, in `held` order — the same
        arithmetic `_mark_position` did per position)."""
        day_prices, out = self.date_lookup[d], {}
        for symbol in self.book.held:
            row = day_prices.loc[symbol] if symbol in day_prices.index else None
            if row is not None:
                self.last_close_by_symbol[symbol] = float(row["close"])
                out[symbol] = float(row["close"])
                continue
            bucket = self.terminal_bucket.get(symbol, "active")
            if bucket == "bankruptcy":
                out[symbol] = 0.0
            elif self.delisting_mode == "lastpx":
                out[symbol] = self.last_close_by_symbol.get(symbol, 0.0)
            elif bucket == "unclassified" and self.delisting_mode == "wipeout_unclassified":
                out[symbol] = 0.0
            else:
                out[symbol] = self.last_close_by_symbol.get(symbol, 0.0)
        return out

    def curve_row(self, d, value: float) -> bool:
        return True

    def after_close(self, d) -> None:
        return None

    def decisions(self) -> list[dict]:
        return self._decisions

    def diagnostics(self) -> dict:
        sd = self.source.diagnostics
        out: dict[str, object] = {
            "split_blocked_name_months": sd.split_blocked_name_months,
            "mania_blocked_name_months": sd.mania_blocked_name_months,
            "terminal_bucket_counts": pd.Series(self.terminal_bucket).value_counts().to_dict(),
            "symbols_after_asset_filter": int(self.px["ticker"].nunique()),
            "cash_shortfall_entry_skips": self.cash_shortfall_count,
        }
        out.update(self.metadata_diag)
        return out
