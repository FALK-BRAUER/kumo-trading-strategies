"""The results tracked beside every variation are what the code in this tree produces from the data in
its set — rerun each variation in a scratch copy of its set and compare `metrics.json` number for number.

This is the reproduction gate for the public evidence (ks#211, ks#240): a runner change that moves a
shipped headline fails HERE, by set, variation and metric, instead of leaving a stale number in the tree.
Runs take 1–15 s each; the whole sweep is ~40 s.

The subprocess is pinned to THIS tree's root — the interpreter's editable install may point at a sibling checkout.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kumo_strategies.backtesting import layout

PKG = Path(__file__).resolve().parents[2]
SETS = sorted(p for p in (PKG / "strategies").glob("*/backtests/*") if p.is_dir() and (p / "variations").is_dir())
VARIATIONS = sorted((s, v) for s in SETS for v in (s / "variations").iterdir() if v.is_dir())

#: Metrics that are timing- or environment-shaped, never compared.
UNSTABLE = set()


def _ids():
    return [f"{s.parent.parent.name}/{s.name}/{v.name}" for s, v in VARIATIONS]


@pytest.mark.parametrize("set_dir,variation", VARIATIONS, ids=_ids())
def test_rerunning_the_variation_reproduces_its_tracked_metrics(set_dir: Path, variation: Path, tmp_path: Path):
    tracked = json.loads((variation / "results" / "metrics.json").read_text())
    scratch = tmp_path / set_dir.name
    shutil.copytree(set_dir, scratch, ignore=shutil.ignore_patterns("results", "report.html", "__pycache__"))
    run = scratch / "variations" / variation.name / "run.py"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PKG.parent), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    proc = subprocess.run([sys.executable, str(run)], capture_output=True, text=True, cwd=str(scratch), timeout=600, env=env)
    assert proc.returncode == 0, f"{run} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    fresh = json.loads((scratch / "variations" / variation.name / "results" / "metrics.json").read_text())
    assert set(fresh) == set(tracked), f"metric set changed: {sorted(set(fresh) ^ set(tracked))}"
    off = {}
    for k, want in tracked.items():
        if k in UNSTABLE:
            continue
        got = fresh[k]
        if isinstance(want, (int, float)) and isinstance(got, (int, float)) and not isinstance(want, bool):
            if not (math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9) or (math.isnan(got) and math.isnan(want))):
                off[k] = (want, got)
        elif got != want:
            off[k] = (want, got)
    assert not off, f"{set_dir.name}/{variation.name}: tracked vs rerun differ — {off}"
    # the eight files, and the provenance names the same inputs
    names = sorted(p.name for p in (scratch / "variations" / variation.name / "results").iterdir())
    assert names == sorted(layout.RESULT_FILES)
