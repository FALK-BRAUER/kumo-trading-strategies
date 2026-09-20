"""Evaluate arms over MANY independent periods instead of one arbitrary split (#30).

WHY THIS REPLACES THE 60/40 SPLIT
---------------------------------
A single split point yields two observations. "Both halves agree" is n=2, and the cut itself is
arbitrary — 60/40 is a convention, and a different cut can reverse the answer. On 2026-08-15 a
schedule comparison was called a failure on exactly that basis, when what the data actually showed
was that the ranking was unstable between two windows. Two windows cannot distinguish "unstable"
from "unlucky".

Many periods give a DISTRIBUTION. An arm that beats the control in 5 of 6 periods is saying
something; 3 of 6 is a coin. The win count is the statistic, not the average return — averages hide
exactly the instability this is built to expose.

Each period is run INDEPENDENTLY, starting flat with fresh capital, rather than by slicing one
continuous equity curve. Slicing carries positions and compounding across the boundary, so an early
period's book contaminates the next one's opening state; independent runs make the periods genuinely
comparable. It also makes them embarrassingly parallel.

THE ARMS-DIFFER GATE APPLIES HERE TOO
-------------------------------------
`run_ablation` refuses to report KPIs for arms that produced byte-identical fills (#26). This
harness was built without that gate and immediately demonstrated why it exists: a PEAK sweep was run
with two arms that had both silently inherited the same `give_back_frac`, so "PEAK" was never
actually tested. The run completed and printed two tidy, slightly-different-looking columns — they
differed only in the label.

The granularity is deliberately "identical in EVERY period", not "identical in one". A rule that
depends on a market state can legitimately be dormant for one quiet window while being live in the
rest; that is a real null, not an inert one. An arm that never once moved a fill across seven
independent periods is not being neutral, it is not connected.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from kumo_strategies.backtesting.ablation import InertArmError, fill_signature


@dataclass(frozen=True)
class ArmSpec:
    """A config PLUS the runner kwargs that arm needs.

    Schedules live in `run_kw` (`slots`), not in the config, so before this existed an arm that
    varied the schedule had to be run in its own `evaluate_periods` call. One arm per call means the
    arms-differ gate has nothing to compare, and a schedule sweep silently opted out of the check
    that exists to catch exactly this class of mistake. It also makes "PEAK at midday" —
    a config change and a schedule change together — expressible as one arm instead of two runs
    whose numbers have to be joined by hand.
    """
    cfg: object
    run_kw: dict


@dataclass(frozen=True)
class Period:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


def calendar_periods(dates: pd.Series, freq: str = "QE", min_sessions: int = 30) -> list[Period]:
    """Non-overlapping calendar periods covering `dates`.

    Periods shorter than `min_sessions` are dropped rather than reported: a 12-session stub produces
    a KPI with enormous variance that then gets counted as one vote alongside a full quarter, which
    silently weights noise equally with evidence.
    """
    d = pd.to_datetime(pd.Series(sorted(pd.unique(dates))))
    out = []
    for stamp, grp in d.groupby(d.dt.to_period(freq[0] if freq[0] in "QMY" else "Q")):
        if len(grp) < min_sessions:
            continue
        out.append(Period(str(stamp), grp.min(), grp.max()))
    return out


def session_blocks(dates: pd.Series, size: int = 40) -> list[Period]:
    """Non-overlapping fixed-length blocks, for when calendar quarters give too few observations.

    Overlapping windows would give more rows but not more INFORMATION — neighbouring windows share
    most of their sessions, so "5 of 6" would be counting the same evidence repeatedly. Blocks stay
    disjoint so each period is one independent vote.
    """
    d = sorted(pd.unique(pd.to_datetime(pd.Series(dates))))
    out = []
    for i in range(0, len(d) - size + 1, size):
        chunk = d[i:i + size]
        out.append(Period(f"b{i//size + 1}:{pd.Timestamp(chunk[0]).date()}",
                          pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1])))
    return out


def _one(args):
    """Top-level so it can be pickled for the process pool.

    The panel arrives already trimmed to `period.end` INCLUDING the warmup history before
    `period.start`, and the runner is told to trade only from `period.start`. Slicing to the period
    alone starves the score — it needs a 60-session vol window and 25 sessions of history, so a
    40-session slice produced 93 trades where the full panel makes 241. That looked like a result
    and was an artefact.
    """
    runner, sub, source, cfg, period, run_kw = args
    res = runner(sub, source, cfg=cfg, trade_from=period.start, **run_kw)
    k = res.report.kpis()
    # Hashed rather than returned whole: the signature only ever gets compared for equality, and
    # shipping every fill back from every worker would dominate the pickling cost of the run.
    sig = pd.util.hash_pandas_object(fill_signature(res), index=False).sum()
    return {"period": period.label, "sessions": int(sub["date"].nunique()), "_sig": int(sig),
            "ret": k["total_return_pct"], "sharpe": k["sharpe"],
            "maxdd": k["max_drawdown_pct"], "trips": k["round_trips"],
            "cost": k.get("total_cost_usd")}


def evaluate_periods(arms: dict, panel: pd.DataFrame, source, runner, periods: list[Period],
                     run_kw: dict, workers: int | None = None) -> pd.DataFrame:
    """One row per (arm, period). `arms` is {label: cfg} or {label: ArmSpec}.

    An `ArmSpec` carries per-arm runner kwargs on top of the shared `run_kw`, which is how a
    schedule arm and a config arm end up in the SAME comparison — and therefore under the same
    arms-differ gate.

    Parallel across the cartesian product, because each run is independent by construction — that
    independence is the same property that makes the periods comparable.
    """
    # Everything up to the period end: the warmup before `start` is scored but not traded.
    sliced = {p.label: panel[panel["date"] <= p.end] for p in periods}
    jobs, keys = [], []
    for label, spec in arms.items():
        cfg = spec.cfg if isinstance(spec, ArmSpec) else spec
        kw = {**run_kw, **spec.run_kw} if isinstance(spec, ArmSpec) else run_kw
        for p in periods:
            jobs.append((runner, sliced[p.label], source, cfg, p, kw))
            keys.append(label)
    n = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=n) as ex:
        rows = list(ex.map(_one, jobs))
    for r, label in zip(rows, keys):
        r["arm"] = label
    df = pd.DataFrame(rows)
    _assert_arms_differ(df)
    return df[["arm", "period", "sessions", "ret", "sharpe", "maxdd", "trips", "cost"]]


def _assert_arms_differ(df: pd.DataFrame) -> None:
    """Refuse to return numbers for two arms that traded identically in every period (#26)."""
    sigs = df.pivot(index="period", columns="arm", values="_sig")
    identical = [(x, y) for x, y in combinations(sigs.columns, 2) if sigs[x].equals(sigs[y])]
    if not identical:
        return
    pairs = "\n".join(f"    {x!r} == {y!r}" for x, y in identical)
    raise InertArmError(
        f"these arms produced byte-identical fills in ALL {len(sigs)} periods:\n{pairs}\n"
        "The parameter separating them is INERT, not neutral. Results are withheld because a null "
        "from an unwired parameter is indistinguishable from a real one. This exact gap already "
        "cost a PEAK sweep: two arms had silently inherited the same give_back_frac, and the run "
        "reported a clean comparison of a rule that was never varied.")


def win_table(df: pd.DataFrame, control: str, metric: str = "ret") -> pd.DataFrame:
    """How often each arm beats the control, period by period. THE headline.

    Reported as a count rather than a mean because the mean is what hid the instability: an arm can
    average well across periods while losing in most of them, carried by one outlier window.
    """
    piv = df.pivot(index="period", columns="arm", values=metric)
    base = piv[control]
    rows = []
    for arm in piv.columns:
        wins = int((piv[arm] > base).sum())
        rows.append({"arm": arm, f"periods_beating_{control}": wins,
                     "of": len(piv), "mean": float(piv[arm].mean()),
                     "worst": float(piv[arm].min()), "best": float(piv[arm].max())})
    return pd.DataFrame(rows).sort_values(f"periods_beating_{control}", ascending=False)
