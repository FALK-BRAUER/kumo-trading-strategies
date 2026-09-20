"""No data file is tracked in this repository (#211).

The data cannot be redistributed; it lives in the private research-data repository and is read
through `KUMO_DATA_ROOT`. The cut is BY FILE TYPE, not by directory — `research/` stays public,
its data files do not. Measured on the cut (2026-09-14): 114 files, 174 MB, and the suite was
byte-for-byte the same with every one of them physically removed (2387 passed / 37 skipped /
17 xfailed), which is the proof nothing here reads them.

Three csvs under `tests/…/fixtures/` stay: ≤30 daily closes for three names each, a test INPUT
rather than a dataset, and the public-tree check exempts `fixtures/` for exactly this. Everything
else of these types, anywhere, is a refusal — and json under `research/` is refused outright,
because json is not a data type by name and every json that was there was data.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_TYPES = re.compile(r"\.(csv|parquet|pkl|npz|h5|feather)$")
FIXTURES = re.compile(r"(^|/)fixtures/")
# A backtest SET carries its own data and results (Falk 2026-09-17, ks#240): parquet via LFS, csv plain.
BACKTEST_SET = re.compile(r"^kumo_strategies/strategies/[^/]+/backtests/[^/]+/")


def tracked() -> list[str]:
    out = subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True)
    return out.split()


def test_no_data_file_is_tracked_outside_a_fixtures_directory() -> None:
    offenders = [f for f in tracked() if DATA_TYPES.search(f) and not FIXTURES.search(f) and not BACKTEST_SET.search(f)]
    assert offenders == [], f"data files tracked outside a fixtures/ or a backtest set: {offenders}"


def test_parquet_in_a_backtest_set_goes_through_lfs() -> None:
    # The set IS the data (ks#240); a 7 MB panel as a plain blob is a repo that never shrinks again.
    import subprocess
    lfs = set(subprocess.check_output(["git", "lfs", "ls-files", "-n"], cwd=ROOT, text=True).split())
    parquet = [f for f in tracked() if BACKTEST_SET.search(f) and f.endswith(".parquet")]
    assert parquet, "no parquet in any backtest set — this test is asserting over nothing"
    assert [f for f in parquet if f not in lfs] == []


def test_no_json_is_tracked_under_research() -> None:
    offenders = [f for f in tracked() if f.startswith("research/") and f.endswith(".json")]
    assert offenders == [], f"json under research/ is data here; move it: {offenders}"


def test_the_fixture_exemption_is_bounded() -> None:
    # The exemption exists for small test inputs. A fixture over 64 KiB is a dataset wearing a hat.
    big = [f for f in tracked() if DATA_TYPES.search(f) and FIXTURES.search(f)
           and (ROOT / f).stat().st_size > 64 * 1024]
    assert big == [], f"fixture files too large to be test inputs: {big}"


def test_git_ignores_every_data_type_under_research_and_data() -> None:
    # The ignore is what stops the NEXT data file from being added by accident; a test that only
    # scans ls-files would catch it one commit late. Asked of git itself (`check-ignore`), not of
    # the .gitignore text — the mechanism, not the prose about it.
    probes = [f"{d}/some-study/probe.{ext}" for d in ("research", "data")
              for ext in ("csv", "parquet", "pkl", "npz", "h5", "feather")]
    probes.append("research/some-study/probe.json")
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", *probes],
                       capture_output=True, text=True)
    ignored = set(r.stdout.split())
    not_ignored = sorted(set(probes) - ignored)
    assert not_ignored == [], f"git would track these data paths: {not_ignored}"


def test_the_fixture_csvs_are_not_ignored_by_accident() -> None:
    # The exemption must survive the ignore rules too, or the next fixture cannot be added.
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index",
                        "tests/strategies/momentum_rotation/fixtures/probe.csv"],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "", "a fixture csv would be ignored"
