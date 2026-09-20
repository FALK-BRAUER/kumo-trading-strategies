"""Every pre-#211 import path still resolves, and to the SAME object as the new one.

The strategy-specific modules left `runtime/` for `strategies/<name>/` (ks#211: everything for a
strategy in one folder). Cockpit imports the old paths — measured on cockpit main, 2–56 sites each —
so each old path is a re-export shim. A shim that re-exports a COPY (a second class object, a
re-evaluated module) would pass an `==` test and still break every `isinstance`, every
`monkeypatch.setattr` through the old path, and every registry keyed on the class; hence `is`.

Seen red first: this file was written before the shims existed, and every parametrized case failed
with ModuleNotFoundError on the old path.
"""

from __future__ import annotations

import importlib

import pytest

from kumo_strategies.strategies._layout import OLD_IMPORT_PATHS

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _public_names(module) -> list[str]:
    """What `from <module> import *` would have exported before the move: every public name, the
    imported ones included — cockpit imports `RiskLimits`, `ADJUSTED` and `make_engine` from
    `pgrunner`, none of which pgrunner defines."""
    return sorted(n for n in vars(module) if not n.startswith("_"))


@pytest.mark.parametrize("old,new", sorted(OLD_IMPORT_PATHS.items()), ids=lambda p: p.rsplit(".", 1)[-1])
def test_the_old_path_is_the_new_object(old: str, new: str):
    new_mod = importlib.import_module(new)
    old_mod = importlib.import_module(old)
    assert old_mod is not new_mod, "a shim is a separate module; an alias in sys.modules is not one"

    names = _public_names(new_mod)
    assert names, f"{new} exports nothing public — the shim would be a shim of nothing"
    missing = [n for n in names if not hasattr(old_mod, n)]
    assert not missing, f"{old} does not re-export {missing}"
    not_same = [n for n in names if getattr(old_mod, n) is not getattr(new_mod, n)]
    assert not_same == [], f"{old} re-exports a DIFFERENT object for {not_same}"

    declared = getattr(old_mod, "__all__", None)
    assert declared is not None, f"{old} declares no __all__ — the re-export list must be explicit"
    assert sorted(declared) == names, (
        f"{old}.__all__ disagrees with what {new} exports:\n"
        f"  only in __all__: {sorted(set(declared) - set(names))}\n"
        f"  only in module : {sorted(set(names) - set(declared))}")


@pytest.mark.parametrize("old,new", sorted(OLD_IMPORT_PATHS.items()), ids=lambda p: p.rsplit(".", 1)[-1])
def test_a_private_name_reaches_through_the_shim(old: str, new: str):
    """`import *` skips underscore names; a test that reaches for `_helper` through the old path
    must still land on the real one, not on AttributeError."""
    new_mod = importlib.import_module(new)
    old_mod = importlib.import_module(old)
    private = sorted(n for n in vars(new_mod)
                     if n.startswith("_") and not n.startswith("__") and not n.endswith("__"))
    if not private:
        pytest.skip(f"{new} defines no private names")
    for n in private:
        assert getattr(old_mod, n) is getattr(new_mod, n), f"{old}.{n} is not {new}.{n}"


def test_the_executor_package_still_exports_the_momentum_runner():
    """`from kumo_strategies.runtime.executor import PgSessionRunner` was a package-level re-export
    (runtime/executor/__init__.py). The runner now lives in strategies/momentum_rotation/runner.py
    and importing it FIRST would cycle through the package `__init__` mid-definition, so the
    re-export is lazy — and must still be the one class."""
    from kumo_strategies.runtime import executor
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner

    assert executor.PgSessionRunner is PgSessionRunner
    assert "PgSessionRunner" in executor.__all__


def test_every_moved_module_is_in_the_table():
    """The table is the ONE list of what moved; a module moved without a row here has no shim test."""
    import pathlib

    from kumo_strategies import strategies

    root = pathlib.Path(strategies.__file__).parent
    on_disk = set()
    for path in root.glob("*/*.py"):
        if path.stem in ("nautilus", "nautilus_intraday", "runner"):
            on_disk.add(f"kumo_strategies.strategies.{path.parent.name}.{path.stem}")
    assert set(OLD_IMPORT_PATHS.values()) == on_disk, (
        f"only on disk: {sorted(on_disk - set(OLD_IMPORT_PATHS.values()))}\n"
        f"only in table: {sorted(set(OLD_IMPORT_PATHS.values()) - on_disk)}")
