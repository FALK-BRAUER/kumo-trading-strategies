"""Where a strategy's runtime code lives, and how to find all of it (ks#211).

Everything for a strategy sits in `strategies/<name>/`: the pure layer (config, engine, exits) AND
its runtime — the Nautilus lane in `nautilus.py` (`nautilus_intraday.py` for the intraday
momentum variant) and, where the lane delegates a session to one, the session runner in
`runner.py`. `runtime/` holds only what is shared between strategies.

THIS MODULE IS THE ONE DISCOVERY POINT. Before the move, ~25 conformance sweeps in `runtime/tests`
each found "every lane" for themselves — `pkgutil.iter_modules(runtime.nautilus)`, or a `glob` over
that directory fed to `ast`. Each was correct while every lane sat in that one directory. After the
move each would have found ZERO lanes, and only eight of them asserted a floor; the other seventeen
would have passed green over nothing. One helper, one floor, every sweep on it.

The floor is EXACT, not `>=`: `tests/.../test_layout.py` pins the set of lane class names, so a lane
that vanishes fails by name and a lane that appears has to be added deliberately.

Discovery is by LOCATION, never by name: a file `strategies/<name>/nautilus.py` is a lane module
because of where it is. A helper that recognised lanes by a naming convention would miss the one that
did not follow it, and nothing would say so.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
from types import ModuleType

__all__ = [
    "LANE_STEMS", "RUNNER_STEMS", "RUNTIME_STEMS", "OLD_IMPORT_PATHS",
    "strategies_root", "strategy_folders",
    "lane_files", "runner_files", "runtime_files",
    "lane_modules", "runner_modules", "runtime_modules",
    "adapters", "lanes", "session_runners",
    "runtime_root", "rel", "is_shim", "shared_nautilus_files", "shared_executor_files",
    "nautilus_sources", "executor_sources", "all_runtime_sources",
]

#: File stems inside `strategies/<name>/` that are RUNTIME, not pure layer. The pure layer must
#: stay importable without nautilus_trader; these are the files allowed to import it.
LANE_STEMS: frozenset[str] = frozenset({"nautilus", "nautilus_intraday"})
RUNNER_STEMS: frozenset[str] = frozenset({"runner"})
RUNTIME_STEMS: frozenset[str] = LANE_STEMS | RUNNER_STEMS

#: Old import path -> new import path. Cockpit imports the old ones (2–56 sites each on cockpit
#: main); every key has a re-export shim at that path and
#: `runtime/tests/test_old_import_paths_are_shims.py` asserts identity through it.
OLD_IMPORT_PATHS: dict[str, str] = {
    "kumo_strategies.runtime.nautilus.bctrot_rotation":
        "kumo_strategies.strategies.bct.nautilus",
    "kumo_strategies.runtime.nautilus.crsi_short":
        "kumo_strategies.strategies.crsi_short.nautilus",
    "kumo_strategies.runtime.nautilus.momentum_rotation":
        "kumo_strategies.strategies.momentum_rotation.nautilus",
    "kumo_strategies.runtime.nautilus.momentum_rotation_intraday":
        "kumo_strategies.strategies.momentum_rotation.nautilus_intraday",
    "kumo_strategies.runtime.nautilus.qc27_rotation":
        "kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus",
    "kumo_strategies.runtime.nautilus.qc345_rotation":
        "kumo_strategies.strategies.qc345_rotation.nautilus",
    "kumo_strategies.runtime.nautilus.smhgld_sleeve":
        "kumo_strategies.strategies.smhgld_sleeve.nautilus",
    "kumo_strategies.runtime.nautilus.template_rotation":
        "kumo_strategies.strategies.template.nautilus",
    "kumo_strategies.runtime.executor.pgrunner":
        "kumo_strategies.strategies.momentum_rotation.runner",
    "kumo_strategies.runtime.executor.qc27_runner":
        "kumo_strategies.strategies.qc27_tech_inverse_vol.runner",
    "kumo_strategies.runtime.executor.template_runner":
        "kumo_strategies.strategies.template.runner",
}


def strategies_root() -> pathlib.Path:
    """`kumo_strategies/strategies/`, from the package actually imported."""
    return pathlib.Path(__file__).resolve().parent


def strategy_folders(root: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Every `strategies/<name>/` package directory, sorted. A folder is a strategy because it is a
    package here; `backtests/`, `tests/` and the like are nested INSIDE one, never beside one."""
    root = root or strategies_root()
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith("_"))


def _files(stems: frozenset[str], root: pathlib.Path | None) -> list[pathlib.Path]:
    return sorted(p for folder in strategy_folders(root)
                  for p in folder.glob("*.py") if p.stem in stems)


def lane_files(root: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Every Nautilus lane module on disk, for the sweeps that read SOURCE (ast) rather than import."""
    return _files(LANE_STEMS, root)


def runner_files(root: pathlib.Path | None = None) -> list[pathlib.Path]:
    return _files(RUNNER_STEMS, root)


def runtime_files(root: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Lanes and runners together: every file under `strategies/` allowed to import nautilus."""
    return _files(RUNTIME_STEMS, root)


def _import(path: pathlib.Path) -> ModuleType:
    root = strategies_root()
    rel = path.resolve().relative_to(root).with_suffix("")
    return importlib.import_module("kumo_strategies.strategies." + ".".join(rel.parts))


def lane_modules() -> list[ModuleType]:
    return [_import(p) for p in lane_files()]


def runner_modules() -> list[ModuleType]:
    return [_import(p) for p in runner_files()]


def runtime_modules() -> list[ModuleType]:
    return [_import(p) for p in runtime_files()]


def adapters() -> list[type]:
    """Every Nautilus `Strategy` subclass DEFINED in a lane module, lane or research, sorted by
    name. A class merely imported into a lane module (a base, a sibling) is that module's business
    and is discovered where it is defined."""
    from nautilus_trader.trading.strategy import Strategy

    out: list[type] = []
    for m in lane_modules():
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ == m.__name__ and issubclass(obj, Strategy) and obj is not Strategy:
                out.append(obj)
    assert out, "no Nautilus adapters discovered under strategies/*/nautilus*.py"
    return sorted(out, key=lambda c: c.__name__)


def lanes() -> list[type]:
    """The adapters cockpit can register: those declaring `EXTERNAL_ID`. Cockpit's registry maps
    external_id -> order_id_tag, so an adapter without one cannot be a lane. Research adapters
    (PennyGapStrategy, IntradayMomentumRotation) have none and are not subject to the deployment
    contract — `test_contract.test_a_research_adapter_cannot_quietly_BECOME_a_lane` closes the hole
    that would open if one grew an `EXTERNAL_ID` without the rest."""
    found = [c for c in adapters() if getattr(c, "EXTERNAL_ID", None)]
    assert found, "no cockpit lanes discovered — this helper no longer describes the package"
    return found


def session_runners() -> list[type]:
    """Every class named `*SessionRunner` defined in a `strategies/<name>/runner.py`."""
    out: list[type] = []
    for m in runner_modules():
        for name, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ == m.__name__ and name.endswith("SessionRunner"):
                out.append(obj)
    assert out, "no session runners discovered under strategies/*/runner.py"
    return sorted(out, key=lambda c: c.__name__)


# --- the shared side: what is left under runtime/ ---------------------------------------------
#
# The sweeps that read SOURCE (ast) used to walk `runtime/nautilus/*.py` or `runtime/**/*.py` and
# thereby covered lanes and shared mixins alike. They still must: a handler defined in a mixin
# (`held_seed.py`, `market_aware.py`) is as much "the order path" as one in a lane. So the file lists
# below are the OLD directory's contents, reassembled: the shared files that stayed, plus the
# strategy files that moved — and never the shims, which are one star-import each and would only
# dilute a scan.


def runtime_root() -> pathlib.Path:
    """`kumo_strategies/runtime/`."""
    return strategies_root().parent / "runtime"


def rel(path: pathlib.Path) -> str:
    """`strategies/momentum_rotation/runner.py` — a path relative to the package, for messages and
    exemption keys. A bare `path.name` no longer identifies a file: every strategy has a
    `nautilus.py`, three have a `runner.py`, and `runtime/executor/runner.py` is a fourth."""
    return path.resolve().relative_to(strategies_root().parent).as_posix()


def is_shim(path: pathlib.Path) -> bool:
    """A re-export shim at a pre-move import path (a key of `OLD_IMPORT_PATHS`)."""
    rel = path.resolve().relative_to(strategies_root().parent.parent).with_suffix("")
    return ".".join(rel.parts) in OLD_IMPORT_PATHS


def _package_files(pkg: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in pkg.glob("*.py") if not is_shim(p))


def shared_nautilus_files() -> list[pathlib.Path]:
    """`runtime/nautilus/*.py` minus the shims: adapter, broker, contract, the mixins."""
    return _package_files(runtime_root() / "nautilus")


def shared_executor_files() -> list[pathlib.Path]:
    """`runtime/executor/*.py` minus the shims (non-recursive: `sources/` and `tests/` are theirs)."""
    return _package_files(runtime_root() / "executor")


def nautilus_sources() -> list[pathlib.Path]:
    """What `runtime/nautilus/*.py` USED to be: the shared adapter layer plus every lane."""
    return shared_nautilus_files() + lane_files()


def executor_sources() -> list[pathlib.Path]:
    """What `runtime/executor/*.py` USED to be: the shared executor plus every session runner."""
    return shared_executor_files() + runner_files()


def all_runtime_sources() -> list[pathlib.Path]:
    """What `runtime/**/*.py` USED to be, tests excluded: everything shared, recursively, plus every
    strategy's lane and runner."""
    shared = sorted(p for p in runtime_root().rglob("*.py")
                    if "tests" not in p.relative_to(runtime_root()).parts
                    and "__pycache__" not in p.parts and not is_shim(p))
    return shared + runtime_files()
