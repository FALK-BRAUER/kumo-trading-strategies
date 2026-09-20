"""Trade history and KPIs that can be checked by hand.

A backtest that reports only a return is unverifiable. This produces the artifacts needed to audit
one: every fill, every round trip, a daily equity curve, and KPIs derived from that curve rather
than asserted alongside it. Anyone can recompute the headline from trades.csv.

Includes the multiple-testing corrections this repo has been ignoring. Roughly 60 configurations
were tried during research; with that many degrees of freedom on ~160 sessions a good Sharpe is
close to guaranteed by chance, so the DEFLATED Sharpe (Bailey & Lopez de Prado) is the number to
read, not the raw one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from kumo_strategies.strategies.give_back import LONG, SHORT, favourable_gain

TRADING_DAYS = 252


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def round_trips(fills: pd.DataFrame) -> pd.DataFrame:
    """FIFO-match opening fills to closing ones into round trips, carrying realised P&L and costs.

    WHICH SIDE OPENS IS DECIDED BY THE BOOK, NOT BY THE WORD "BUY". A fill closes the position when
    one is open on the other side, and opens one otherwise. That is how the tape reads, it leaves
    every long lane's output byte-identical (a BUY always opens when flat), and it is the ONLY way a
    short book can be reported here at all.

    Reading BUY as "open" unconditionally does not fail on a short book — it drops the opening SELL
    silently and then INVERTS the sign of every return it does report. That is the `sides.py` defect
    in reporting form: the dangerous version is the one that succeeds. #123's headline is a mean of
    +6.04% a trade and a worst trade of -53%; both come out with the wrong sign.
    """
    if fills.empty:
        return pd.DataFrame()
    f = fills.sort_values("ts").reset_index(drop=True)
    open_lots: dict[str, list[dict]] = {}
    out = []
    for _, t in f.iterrows():
        sym, side = t["symbol"], t["side"]
        lots = open_lots.setdefault(sym, [])
        if not lots or lots[0]["side"] == side:
            lots.append({"ts": t["ts"], "px": t["price"], "qty": t["qty"], "cost": t["cost"],
                         "side": side})
            continue
        remaining = t["qty"]
        while remaining > 1e-9 and lots:
            lot = lots[0]
            take = min(remaining, lot["qty"])
            share = take / t["qty"] if t["qty"] else 0.0
            lot_share = take / lot["qty"] if lot["qty"] else 0.0
            long_side = lot["side"] == "BUY"
            # Signed in the POSITION's direction, from the one module that owns that convention:
            # a short at 100 covered at 90 is +10%, and P&L follows the same sign.
            gain = favourable_gain(lot["px"], t["price"], side=LONG if long_side else SHORT)
            gross = (t["price"] - lot["px"]) * take * (1.0 if long_side else -1.0)
            cost = lot["cost"] * lot_share + t["cost"] * share
            out.append({
                "symbol": sym, "side": "LONG" if long_side else "SHORT",
                "entry_ts": lot["ts"], "exit_ts": t["ts"],
                "entry_px": lot["px"], "exit_px": t["price"], "qty": take,
                "gross_pnl": gross, "cost": cost, "net_pnl": gross - cost,
                "return_pct": 100.0 * gain,
                "hold_days": (t["ts"] - lot["ts"]).total_seconds() / 86400.0,
            })
            lot["qty"] -= take
            lot["cost"] *= (1 - lot_share)
            remaining -= take
            if lot["qty"] <= 1e-9:
                lots.pop(0)
        if remaining > 1e-9:                     # the fill flipped the book through flat
            lots.append({"ts": t["ts"], "px": t["price"], "qty": remaining,
                         "cost": t["cost"] * (remaining / t["qty"] if t["qty"] else 0.0),
                         "side": side})
    return pd.DataFrame(out)


def deflated_sharpe(sr: float, n: int, n_trials: int, skew: float, kurt: float) -> float:
    """Bailey & Lopez de Prado: probability the true Sharpe exceeds zero, after correcting for the
    number of configurations tried and for non-normal returns. Reported as a probability."""
    if n < 20 or not np.isfinite(sr):
        return float("nan")
    # expected maximum Sharpe from n_trials independent draws of a zero-skill strategy
    e = 0.5772156649
    if n_trials > 1:
        z = math.sqrt(2 * math.log(n_trials))
        sr0 = (1 - e) * z + e * (z - (math.log(math.log(n_trials)) + math.log(4 * math.pi)) / (2 * z))
        sr0 /= math.sqrt(TRADING_DAYS)                    # per-period, matching sr's units below
    else:
        sr0 = 0.0
    num = (sr - sr0) * math.sqrt(n - 1)
    den = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    from statistics import NormalDist
    return float(NormalDist().cdf(num / den))


@dataclass
class Report:
    equity: pd.Series
    trades: pd.DataFrame
    fills: pd.DataFrame
    starting_cash: float
    n_trials: int = 1
    benchmarks: dict[str, pd.Series] = field(default_factory=dict)

    def active_equity(self) -> pd.Series:
        """Equity from the first fill onward.

        The candidate pool is empty before the source has coverage, so a panel that starts earlier
        gives a flat equity prefix. Those sessions are not 'zero-return days the strategy chose' —
        the strategy did not exist yet — but they still enter the volatility and Sharpe denominators
        and drag every risk metric down (1.94 -> 1.22 on this run). Measure the traded period.
        """
        e = self.equity.dropna()
        if self.fills is None or not len(self.fills):
            return e
        first = pd.Timestamp(pd.to_datetime(self.fills["ts"]).min()).normalize()
        return e[e.index >= first]

    def kpis(self) -> dict:
        e = self.active_equity()
        r = e.pct_change().dropna()
        if len(r) < 2:
            return {"error": "insufficient equity history"}
        base = float(e.iloc[0]) if len(e) else self.starting_cash
        total = e.iloc[-1] / base - 1.0
        years = len(r) / TRADING_DAYS
        dd = e / e.cummax() - 1.0
        downside = r[r < 0]
        sr_p = r.mean() / r.std() if r.std() else float("nan")     # per-period Sharpe
        t = self.trades
        wins = t[t.net_pnl > 0] if len(t) else t
        losses = t[t.net_pnl <= 0] if len(t) else t
        gross_win = wins.net_pnl.sum() if len(wins) else 0.0
        gross_loss = abs(losses.net_pnl.sum()) if len(losses) else 0.0
        k = {
            "sessions_traded": len(e),
            "sessions_in_panel": len(self.equity.dropna()),
            "start": str(e.index.min().date()), "end": str(e.index.max().date()),
            "total_return_pct": 100 * _safe(total),
            "cagr_pct": 100 * _safe((1 + total) ** (1 / years) - 1) if years > 0 else float("nan"),
            "ann_vol_pct": 100 * _safe(r.std() * math.sqrt(TRADING_DAYS)),
            "sharpe": _safe(sr_p * math.sqrt(TRADING_DAYS)),
            "sortino": _safe(r.mean() / downside.std() * math.sqrt(TRADING_DAYS)) if len(downside) > 1 else float("nan"),
            "max_drawdown_pct": 100 * _safe(dd.min()),
            "calmar": _safe((total / years) / abs(dd.min())) if dd.min() < 0 and years > 0 else float("nan"),
            "longest_drawdown_days": int(self._longest_dd(dd)),
            "skew": _safe(r.skew()), "kurtosis": _safe(r.kurtosis() + 3.0),
            "best_day_pct": 100 * _safe(r.max()), "worst_day_pct": 100 * _safe(r.min()),
            "positive_days_pct": 100 * _safe((r > 0).mean()),
            "n_trials_corrected_for": self.n_trials,
            "deflated_sharpe_prob": _safe(deflated_sharpe(
                sr_p, len(r), self.n_trials, float(r.skew()), float(r.kurtosis() + 3.0))),
            "round_trips": len(t),
            "win_rate_pct": 100 * _safe(len(wins) / len(t)) if len(t) else float("nan"),
            "avg_win_pct": _safe(wins.return_pct.mean()) if len(wins) else float("nan"),
            "avg_loss_pct": _safe(losses.return_pct.mean()) if len(losses) else float("nan"),
            "profit_factor": _safe(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "avg_hold_days": _safe(t.hold_days.mean()) if len(t) else float("nan"),
            "total_cost_usd": _safe(self.fills.cost.sum()) if len(self.fills) else 0.0,
            "cost_pct_of_capital": 100 * _safe(self.fills.cost.sum() / base) if len(self.fills) else 0.0,
            "gross_return_pct": 100 * _safe(total + self.fills.cost.sum() / base) if len(self.fills) else 100 * _safe(total),
            "n_fills": len(self.fills),
            "turnover_usd": _safe((self.fills.qty * self.fills.price).abs().sum()) if len(self.fills) else 0.0,
        }
        for name, series in self.benchmarks.items():
            b = series.reindex(e.index).dropna()
            if len(b) < 2:
                continue
            br = b.pct_change().dropna()
            bdd = b / b.cummax() - 1.0
            k[f"bench_{name}_return_pct"] = 100 * _safe(b.iloc[-1] / b.iloc[0] - 1)
            k[f"bench_{name}_sharpe"] = _safe(br.mean() / br.std() * math.sqrt(TRADING_DAYS))
            k[f"bench_{name}_maxdd_pct"] = 100 * _safe(bdd.min())
            common = r.index.intersection(br.index)
            if len(common) > 10:
                k[f"alpha_vs_{name}_pct"] = k["total_return_pct"] - k[f"bench_{name}_return_pct"]
                k[f"corr_to_{name}"] = _safe(r.reindex(common).corr(br.reindex(common)))
        return k

    @staticmethod
    def _longest_dd(dd: pd.Series) -> int:
        run = best = 0
        for v in dd:
            run = run + 1 if v < -1e-9 else 0
            best = max(best, run)
        return best

    def monthly(self) -> pd.DataFrame:
        e = self.active_equity()
        m = e.resample("ME").last()
        out = (m.pct_change() * 100).to_frame("return_pct")
        out.iloc[0, 0] = (m.iloc[0] / float(e.iloc[0]) - 1) * 100
        return out

    def write(self, outdir: str | Path) -> Path:
        import json
        p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
        self.fills.to_csv(p / "fills.csv", index=False)
        self.trades.to_csv(p / "trades.csv", index=False)
        self.equity.rename("equity").to_frame().to_csv(p / "equity.csv")
        self.monthly().to_csv(p / "monthly.csv")
        json.dump(self.kpis(), open(p / "kpis.json", "w"), indent=1, default=str)
        return p
