"""The discovery helper finds EXACTLY the lanes and runners that ship (ks#211).

Every conformance sweep in `runtime/tests` takes "every lane" from `strategies/_layout.py`. Before
the move each sweep found the lanes for itself by looking in `runtime/nautilus/`; after the move that
directory holds shims, and a sweep still looking there finds nothing and passes over nothing —
seventeen of twenty-five had no floor. This is the ONE floor, and it is exact: a lane that vanishes
fails here BY NAME, and a lane that appears has to be added here deliberately.

Seen red first: with `LANE_STEMS` emptied (the helper pointed at nothing) every test here failed —
`adapters()` on its own assertion, the pinned sets on the missing names.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from kumo_strategies.strategies import _layout as L

#: The adapters that ship, by class name. Research adapters (no EXTERNAL_ID) included: they are
#: still Nautilus strategies in the package and still subject to the sweeps that apply to all.
ADAPTERS = {
    "BCTRotationStrategy", "CrsiShortStrategy", "IntradayMomentumRotation",
    "MomentumRotationStrategy", "QC27RotationStrategy",
    "QC345RotationStrategy", "SmhGldSleeveStrategy", "TemplateRotationStrategy",
}
#: The cockpit-registrable subset: class name -> EXTERNAL_ID.
LANES = {
    "BCTRotationStrategy": "BCTROT", "CrsiShortStrategy": "CRSISHORT",
    "MomentumRotationStrategy": "MOMENTUM", "QC27RotationStrategy": "QC27",
    "QC345RotationStrategy": "QC345", "SmhGldSleeveStrategy": "SMHGLD",
    "TemplateRotationStrategy": "TEMPLATE",
}
RUNNERS = {"PgSessionRunner", "QC27SessionRunner", "TemplateSessionRunner"}
#: Every strategy folder, and which runtime files each holds. A folder with no lane is allowed
#: (a pure research strategy) but has to be listed as such, so a lane file that goes missing is a
#: named failure rather than a shorter list.
FOLDERS = {
    "bct": {"nautilus.py"},
    "crsi_short": {"nautilus.py"},
    "momentum_rotation": {"nautilus.py", "nautilus_intraday.py", "runner.py"},
    "qc27_tech_inverse_vol": {"nautilus.py", "runner.py"},
    "qc345_rotation": {"nautilus.py"},
    "smhgld_sleeve": {"nautilus.py"},
    "template": {"nautilus.py", "runner.py"},
}


def test_the_folders_and_their_runtime_files_are_exactly_these():
    found = {f.name: {p.name for p in f.glob("*.py") if p.stem in L.RUNTIME_STEMS}
             for f in L.strategy_folders()}
    assert found == FOLDERS


def test_every_adapter_is_discovered_by_name():
    assert {c.__name__ for c in L.adapters()} == ADAPTERS


def test_every_lane_is_discovered_with_its_external_id():
    assert {c.__name__: c.EXTERNAL_ID for c in L.lanes()} == LANES


def test_every_session_runner_is_discovered_by_name():
    assert {c.__name__ for c in L.session_runners()} == RUNNERS


def test_a_lane_is_defined_in_its_own_folder():
    """`adapters()` filters on `__module__`, so a lane imported into a sibling module (BCT extends
    MOMENTUM's lane) is counted once, where it is defined."""
    for cls in L.adapters():
        parts = cls.__module__.split(".")
        assert parts[:2] == ["kumo_strategies", "strategies"], cls.__module__
        assert parts[-1] in L.LANE_STEMS, cls.__module__


def test_the_helper_finds_nothing_in_an_empty_tree(tmp_path):
    """The floor above is only a floor if an empty tree yields an empty answer rather than a
    fallback to the real one."""
    assert L.strategy_folders(tmp_path) == []
    assert L.lane_files(tmp_path) == []
    assert L.runner_files(tmp_path) == []


@pytest.mark.parametrize("folder", sorted(FOLDERS), ids=str)
def test_the_pure_layer_imports_without_nautilus(folder: str):
    """Importing `kumo_strategies.strategies.<name>` must not load nautilus_trader. The lane now
    sits in the same folder as the engine, so the boundary is no longer a directory; it is that the
    package `__init__` never imports its own `nautilus.py`/`runner.py`. A fresh interpreter, because
    this process has long since imported nautilus."""
    code = (f"import sys; import kumo_strategies.strategies.{folder}; "
            f"bad = sorted(m for m in sys.modules if m.startswith('nautilus_trader')); "
            f"sys.exit(1 if bad else 0)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, (
        f"importing strategies.{folder} pulled nautilus_trader into the process:\n{done.stderr[-2000:]}")


@pytest.mark.parametrize("path", L.strategy_folders(), ids=lambda p: p.name)
def test_no_pure_module_imports_nautilus(path):
    """Statically: only the runtime files in a folder may name nautilus in an import."""
    import ast
    for f in sorted(path.glob("*.py")):
        if f.stem in L.RUNTIME_STEMS:
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            hits = [n for n in names if "nautilus" in n.lower()]
            assert not hits, f"{f.relative_to(L.strategies_root())} imports {hits} — the pure layer must stay runtime-free"
