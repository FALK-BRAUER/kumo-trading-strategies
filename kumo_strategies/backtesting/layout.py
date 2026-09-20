"""One layout for every backtest (ks#240, revised by Falk 2026-09-17).

    strategies/<name>/backtests/<set>/
        bars.parquet, ledger.csv, …          the data — the set IS the data; git's blob hash is its provenance
        report.html                          one row per variation, metrics side by side, equity curves overlaid
        variations/<variant>/
            run.py                           reads ../../<data>; loads parameters.json; runs; writes results/
            parameters.json                  every parameter the run used — nothing inline in run.py
            results/                         exactly RESULT_FILES, tracked beside the run that made them

No dataset.json, no KUMO_DATA_ROOT indirection: a set is self-contained. What `write_results` writes is
the whole contract — exactly `RESULT_FILES`, no more (a stray file is a result nobody named), no fewer
(a missing one is a claim the run did not make). `provenance.json` records the git sha of the tree,
the sha256 of every input file the run named and of parameters.json, and the package versions.
"""
from __future__ import annotations

import hashlib
import html
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kumo_strategies.backtesting.report import Report

RESULT_FILES: tuple[str, ...] = (
    "factsheet.html", "metrics.json", "fills.csv", "trades.csv", "equity.csv", "monthly.csv",
    "diagnostics.json", "provenance.json",
)
VARIATION_FILES: tuple[str, ...] = ("run.py", "parameters.json")


def set_dir(variation_dir: Path) -> Path:
    """`<set>/` for a `<set>/variations/<variant>/`."""
    return Path(variation_dir).resolve().parent.parent


def load_parameters(variation_dir: Path) -> dict:
    return json.loads((Path(variation_dir) / "parameters.json").read_text())


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str).encode()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version(), "pandas": pd.__version__}
    for dist, key in (("kumo-trading-strategies", "kumo_strategies"), ("nautilus_trader", "nautilus_trader")):
        try:
            out[key] = version(dist)
        except PackageNotFoundError:
            out.setdefault(key, "unknown")
    return out


def write_results(variation_dir: Path, report: Report, *, decisions: pd.DataFrame, parameters: dict,
                  inputs: list[Path], diagnostics: dict | None = None) -> Path:
    """The eight files, and only them, into `<variation>/results/`. `inputs` are the data files the run
    read; each is hashed into provenance so a result is tied to exactly the bytes that produced it."""
    p = Path(variation_dir) / "results"
    p.mkdir(parents=True, exist_ok=True)
    report.fills.to_csv(p / "fills.csv", index=False)
    report.trades.to_csv(p / "trades.csv", index=False)
    report.equity.rename("equity").to_frame().to_csv(p / "equity.csv")
    monthly = report.monthly()
    monthly.to_csv(p / "monthly.csv")
    metrics = json.loads(json.dumps(report.kpis(), default=str))
    (p / "metrics.json").write_text(json.dumps(metrics, indent=1))
    diag = {"decisions": decisions.to_dict(orient="records") if len(decisions) else []}
    diag.update(diagnostics or {})
    (p / "diagnostics.json").write_text(json.dumps(diag, indent=1, default=str))
    sd = set_dir(variation_dir)
    prov = {
        "generated_at": _now(), "git_sha": _git_sha(), "package": _package_versions(),
        "strategy": _strategy_of(sd), "set": sd.name, "variation": Path(variation_dir).name,
        "inputs": [{"path": str(Path(i).resolve().relative_to(sd)), "sha256": _sha256_file(Path(i)),
                    "bytes": Path(i).stat().st_size} for i in inputs],
        "parameters_sha256": _sha256_bytes(_canon(parameters)), "argv": sys.argv,
    }
    (p / "provenance.json").write_text(json.dumps(prov, indent=1))
    # The page charts ACTIVE equity — from the first fill, the same series the KPIs are computed on — so
    # a warm-up prefix (QC345's 252-session lookback) does not read as a year of underperformance.
    (p / "factsheet.html").write_text(factsheet_html(metrics, monthly, report.active_equity(), parameters, prov,
                                                      benchmarks=report.benchmarks, trades=report.trades))
    stray = sorted(f.name for f in p.iterdir() if f.name not in RESULT_FILES)
    if stray:
        raise RuntimeError(f"results/ holds files the layout does not name: {stray}")
    return p


def _strategy_of(set_path: Path) -> str:
    """`strategies/<name>/backtests/<set>/` → `<name>`; the folder above `backtests/` is the strategy."""
    sd = Path(set_path)
    return sd.parent.parent.name if sd.parent.name == "backtests" else sd.parent.name


def factsheet_html(metrics: dict, monthly, equity: pd.Series, parameters: dict, prov: dict, *,
                   benchmarks: dict[str, pd.Series] | None = None, trades: pd.DataFrame | None = None) -> str:
    """One self-contained page — no scripts, no external resources. Rendered by `factsheet.py`."""
    from kumo_strategies.backtesting.factsheet import factsheet_page
    return factsheet_page(metrics, monthly, equity, parameters, prov, benchmarks=benchmarks, trades=trades)


def write_set_report(set_path: Path) -> Path:
    """`<set>/report.html`: every variation that has results, side by side — what each IS, the headline
    numbers, the curves overlaid in fixed slot colours, returns by year, every metric. Rendered by
    `factsheet.py`."""
    from kumo_strategies.backtesting.factsheet import set_report_page
    sd = Path(set_path)
    rows, curves, whats = {}, {}, {}
    for v in sorted(p for p in (sd / "variations").iterdir() if p.is_dir()):
        r = v / "results"
        if not (r / "metrics.json").is_file():
            continue
        rows[v.name] = json.loads((r / "metrics.json").read_text())
        curves[v.name] = pd.read_csv(r / "equity.csv", index_col=0, parse_dates=True)["equity"]
        whats[v.name] = load_parameters(v).get("variant", "") if (v / "parameters.json").is_file() else ""
    out = sd / "report.html"
    out.write_text(set_report_page(_strategy_of(sd), sd.name, rows, curves, whats, _now()))
    return out
