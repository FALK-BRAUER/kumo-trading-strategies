"""Run a set of arms and refuse to report KPIs for any arm that is not distinguishable (#26).

WHY THIS IS A HARNESS AND NOT A HELPER
--------------------------------------
Two `PortfolioConfig` flags have been swept in this repo while having no effect on results, and the
second survived a fix of the first:

  `max_correlation`     `decide()` was called without `corr`, so `_diversified` short-circuited.
                        Sweeping 0.7 and 0.85 returned byte-identical numbers to baseline.
  `inverse_vol_sizing`  the same missing `vol`, AND — after that was fixed — a runner that computed
                        `dec.weights` and sized every position at a flat `base * deployed / n_hold`,
                        never reading them.

A sweep that cannot distinguish its arms does not fail. It returns a clean, plausible "no effect"
for a mechanism that was never running: every arm agrees, the variance looks reassuringly low, and
the conclusion reads "this does not help" when the truth is "this was never tried". A null is the
one result that inertness perfectly imitates.

So the check cannot be a helper each variant remembers to call — that is the same failure one level
up, and #24 has five more size-varying variants (F1, F3, F5, F6, G) queued to rediscover it. Here it
is structural: `run_ablation` raises before returning, so there is no path from a config to a KPI
that skips it.

    from kumo_strategies.backtesting.ablation import Arm, run_ablation

    arms = [Arm("A control", cfg_a), Arm("B", cfg_b)]
    ab = run_ablation(arms, panel, source, instruments_path=..., cost_model=...)
    print(ab.kpi_frame())          # unreachable unless every arm differed

`InertArmError` is deliberately not catchable-by-convention — there is no `allow_inert` flag. An arm
that legitimately needs to be identical to another is not an ablation arm, and calling
`runner_verified.run` directly is the honest way to say so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from kumo_strategies.backtesting.report import Report, round_trips
from kumo_strategies.backtesting.runner_sessions import BacktestResult  # the ONE runner (#270)
from kumo_strategies.backtesting.runner_sessions import run_sessions as run
from kumo_strategies.strategies.momentum_rotation.candidates import CandidateSource
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

#: The columns that make two runs comparable. Costs and venue are derived from these, so two runs
#: agreeing here traded identically in every way that can move a KPI.
SIGNATURE_COLS = ("ts", "symbol", "side", "qty", "price")


class InertArmError(RuntimeError):
    """An arm produced fills identical to another arm — the parameter is inert, not neutral."""


@dataclass(frozen=True)
class Arm:
    label: str
    cfg: MomentumRotationConfig


@dataclass(frozen=True)
class ArmResult:
    label: str
    cfg: MomentumRotationConfig
    result: BacktestResult


def fill_signature(result: BacktestResult) -> pd.DataFrame:
    """The comparable part of a run. Two arms agreeing here placed the same trades."""
    f = result.report.fills
    if not len(f):
        return pd.DataFrame(columns=list(SIGNATURE_COLS))
    return (f[list(SIGNATURE_COLS)]
            .sort_values(["ts", "symbol", "side"])
            .reset_index(drop=True))


def deployed_fraction(result: BacktestResult) -> float:
    """Mean fraction of equity actually at risk.

    Worth reporting on every arm because backtest and live bound the book differently: the backtest
    has no `max_positions` check at all and is bounded by `n_hold` inside `decide()`, while the live
    runner re-checks a cap counting positions. An arm can look well-deployed here and under-deploy
    live for a reason this harness never models.

    It is also how a size-varying arm hides a cash drag inside what looks like a risk effect —
    inverse-vol sizing moved deployment 63.3% -> 59.4% with the traded name set completely
    unchanged, and that shortfall landed in the return looking exactly like the risk tilt.
    """
    f = result.report.fills
    eq = result.report.active_equity()
    if not len(f) or not len(eq):
        return float("nan")
    signed = np.where(f.side == "BUY", f.qty * f.price, -(f.qty * f.price))
    invested = (pd.Series(signed, index=pd.to_datetime(f.ts).dt.normalize())
                .groupby(level=0).sum().cumsum().reindex(eq.index, method="ffill").fillna(0.0))
    return float((invested / eq).mean())


@dataclass
class Ablation:
    arms: list[ArmResult]

    def __getitem__(self, label: str) -> BacktestResult:
        for a in self.arms:
            if a.label == label:
                return a.result
        raise KeyError(label)

    @property
    def labels(self) -> list[str]:
        return [a.label for a in self.arms]

    def traded_names(self) -> dict[str, set[str]]:
        return {a.label: (set(a.result.report.fills.symbol)
                          if len(a.result.report.fills) else set()) for a in self.arms}

    def kpi_frame(self) -> pd.DataFrame:
        """One row per arm, on the FULL sample. Return, Sharpe and drawdown together.

        Read `walk_forward_frame()` before trusting anything here. A full-sample sweep certified a
        curve fit on 2026-08-15 — the aggregate was monotone while both halves zigzagged — so these
        numbers are the summary, not the evidence.
        """
        rows = []
        for a in self.arms:
            k = a.result.report.kpis()
            rows.append({"arm": a.label,
                         "deployed_pct": 100 * deployed_fraction(a.result),
                         **{c: k.get(c) for c in (
                             "total_return_pct", "sharpe", "max_drawdown_pct",
                             "deflated_sharpe_prob", "round_trips", "win_rate_pct",
                             "profit_factor", "avg_hold_days", "total_cost_usd")}})
        return pd.DataFrame(rows)

    def walk_forward_frame(self, split: float = 0.6) -> pd.DataFrame:
        """Per-arm KPIs on each half of the sample, not the aggregate.

        THIS IS THE ONE TO READ FIRST, and it exists because the aggregate lies. On 2026-08-15 a
        give-back sweep looked cleanly monotone across the full panel — Sharpe falling 2.02 to 1.50
        as the parameter widened — and was called the strongest result of the day. Split in two, the
        monotonicity appeared in NEITHER half; both zigzagged, and the ordering of the halves
        disagreed. Aggregation had manufactured the smoothness.

        The band criterion — a coherent region of good values rather than one good cell — is the only
        reliable noise detector available on ~160 in-sample sessions, and it has to be applied to
        each half. Applied to the aggregate it certified a curve fit.

        `split` is a fraction of sessions and should be fixed BEFORE any result is seen. Choosing it
        afterwards fits the split itself, which is the failure this method exists to catch.
        """
        rows = []
        for a in self.arms:
            eq = a.result.report.active_equity()
            if len(eq) < 40:
                continue
            cut = eq.index[int(len(eq) * split)]
            row = {"arm": a.label}
            fills = a.result.report.fills
            ft = pd.to_datetime(fills.ts) if len(fills) else None
            for tag, mask in (("is", eq.index <= cut), ("oos", eq.index > cut)):
                e = eq[mask]
                f = (fills[ft <= cut] if tag == "is" else fills[ft > cut]) if ft is not None \
                    else fills
                sub = Report(equity=e, fills=f, trades=round_trips(f) if len(f) else pd.DataFrame(),
                             starting_cash=float(e.iloc[0]), n_trials=1)
                k = sub.kpis()
                row[f"{tag}_ret"] = k["total_return_pct"]
                row[f"{tag}_sharpe"] = k["sharpe"]
                row[f"{tag}_maxdd"] = k["max_drawdown_pct"]
                row[f"{tag}_trips"] = k["round_trips"]
            row["sign_agrees"] = (row["is_ret"] > 0) == (row["oos_ret"] > 0)
            rows.append(row)
        return pd.DataFrame(rows)

    def population_report(self, control: str | None = None) -> pd.DataFrame:
        """Which names each arm traded, against the control.

        The ticket standard is "same population across arms, asserted not assumed". Asserting
        EQUALITY would be wrong — a size-varying arm legitimately changes which entries survive the
        cash check, and a correlation cap legitimately changes which name fills a slot. So this
        reports the divergence rather than forbidding it: a KPI comparison across different name
        sets is not a comparison, and the reader has to be able to see when that has happened.
        """
        pops = self.traded_names()
        base = control or self.labels[0]
        ref = pops[base]
        return pd.DataFrame([{"arm": lab, "n_names": len(p),
                              "only_here": len(p - ref), "missing_vs_control": len(ref - p),
                              "only_here_names": ",".join(sorted(p - ref)),
                              "missing_names": ",".join(sorted(ref - p))}
                             for lab, p in pops.items()])


def _assert_arms_differ(arms: Sequence[ArmResult]) -> None:
    sigs = {a.label: fill_signature(a.result) for a in arms}
    identical = [(x, y) for x, y in combinations(sigs, 2) if sigs[x].equals(sigs[y])]
    if not identical:
        return
    pairs = "\n".join(f"    {x!r} == {y!r}" for x, y in identical)
    raise InertArmError(
        "these arms produced byte-identical fills:\n"
        f"{pairs}\n"
        "The parameter that separates them is INERT, not neutral — it is not connected to anything "
        "that can move a fill. KPIs are being withheld because a null from an inert parameter is "
        "indistinguishable from a real null, and reporting one would certify a mechanism that never "
        "ran. Check that the config reaches decide(), and that whatever decide() returns is actually "
        "consumed by the sizing or selection path.")


def run_ablation(arms: Sequence[Arm], panel: pd.DataFrame, source: CandidateSource,
                 *, on_arm: Any = None, **run_kw: Any) -> Ablation:
    """Run every arm on the same panel, then refuse to return unless they are distinguishable.

    `run_kw` goes to `runner_verified.run` unchanged and is shared by every arm, so the panel,
    costs, instruments and starting capital cannot drift between them — a difference in any of
    those would be attributed to the parameter under test.

    `n_trials` defaults to the number of arms, so the deflated Sharpe corrects for this ablation.
    It does NOT correct for configurations tried across the wider research programme; that remains
    the caller's caveat to state.
    """
    run_kw.setdefault("n_trials", len(arms))
    out = []
    for arm in arms:
        res = run(panel, source, cfg=arm.cfg, **run_kw)
        out.append(ArmResult(label=arm.label, cfg=arm.cfg, result=res))
        if on_arm is not None:
            on_arm(arm.label)
    _assert_arms_differ(out)
    return Ablation(arms=out)
