"""One layout for every backtest (ks#240, revised 2026-09-17): `strategies/<name>/backtests/<set>/` holds
the data, `report.html`, and `variations/<variant>/{run.py, parameters.json, results/}` — eight named result
files, tracked beside the run that made them. No dataset.json, no data-root indirection.

Driven through the real writer on a synthetic Report (built with the real trade matcher so the double
carries the columns kpis() reads). Pinned: the exact result file set; metrics.json IS the report's kpis;
provenance hashes every input file and parameters.json and names the tree; the factsheet and the set
report are self-contained pages (asserted through html.parser, never by substring on the source); the
tree layout holds for every set in the repo.
"""
from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

from kumo_strategies.backtesting import layout
from kumo_strategies.backtesting.report import Report, round_trips

PKG = Path(__file__).resolve().parents[2]          # kumo_strategies/
ROOT = PKG.parent


def _report(scale: float = 37.5) -> Report:
    idx = pd.date_range("2026-01-02", periods=40, freq="B")
    eq = pd.Series(100_000 + (pd.Series(range(40)) * scale).values, index=idx)
    fills = pd.DataFrame({"ts": [idx[1], idx[10]], "symbol": ["AAA", "AAA"], "side": ["BUY", "SELL"],
                          "qty": [10, 10], "price": [100.0, 105.0], "cost": [0.5, 0.5], "venue": ["X", "X"]})
    return Report(equity=eq, trades=round_trips(fills), fills=fills, starting_cash=100_000.0)


class _Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.tags: set[str] = set(); self.text: list[str] = []; self.links: list[str] = []
        self.classes: set[str] = set()
    def handle_starttag(self, tag, attrs):
        self.tags.add(tag); self.links += [v for k, v in attrs if k in ("src", "href") and v]
        self.classes |= {c for k, v in attrs if k == "class" and v for c in v.split()}
    def handle_data(self, data):
        self.text.append(data)


def _set(tmp: Path, variants=("baseline",)) -> Path:
    sd = tmp / "strategies" / "x" / "backtests" / "s1"
    (sd / "variations").mkdir(parents=True)
    (sd / "bars.parquet").write_bytes(b"PAR1 synthetic")
    for v in variants:
        (sd / "variations" / v).mkdir()
        (sd / "variations" / v / "parameters.json").write_text(json.dumps({"n_hold": 8, "variant": v}))
        (sd / "variations" / v / "run.py").write_text("# synthetic\n")
    return sd


def test_write_results_produces_exactly_the_eight_files_inside_the_variation(tmp_path: Path) -> None:
    sd = _set(tmp_path)
    v = sd / "variations" / "baseline"
    out = layout.write_results(v, _report(), decisions=pd.DataFrame({"date": ["2026-01-02"], "action": ["hold"]}),
                               parameters=layout.load_parameters(v), inputs=[sd / "bars.parquet"])
    assert out == v / "results"
    assert sorted(p.name for p in out.iterdir()) == sorted(layout.RESULT_FILES)
    assert set(layout.RESULT_FILES) == {"factsheet.html", "metrics.json", "fills.csv", "trades.csv",
                                        "equity.csv", "monthly.csv", "diagnostics.json", "provenance.json"}


def test_metrics_json_is_the_reports_kpis(tmp_path: Path) -> None:
    sd = _set(tmp_path); r = _report()
    out = layout.write_results(sd / "variations" / "baseline", r, decisions=pd.DataFrame(), parameters={}, inputs=[])
    assert json.load(open(out / "metrics.json")) == json.loads(json.dumps(r.kpis(), default=str))


def test_provenance_hashes_every_input_and_the_parameters_and_names_the_tree(tmp_path: Path) -> None:
    sd = _set(tmp_path); v = sd / "variations" / "baseline"; params = {"n_hold": 8}
    out = layout.write_results(v, _report(), decisions=pd.DataFrame(), parameters=params, inputs=[sd / "bars.parquet"])
    prov = json.load(open(out / "provenance.json"))
    assert prov["set"] == "s1" and prov["variation"] == "baseline"
    assert prov["inputs"] == [{"path": "bars.parquet", "sha256": hashlib.sha256(b"PAR1 synthetic").hexdigest(), "bytes": 14}]
    assert prov["parameters_sha256"] == hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    assert len(prov["git_sha"]) >= 7 and prov["package"]["pandas"] == pd.__version__


def test_diagnostics_carry_the_decisions_and_the_extras(tmp_path: Path) -> None:
    sd = _set(tmp_path); dec = pd.DataFrame({"date": ["2026-01-02"], "action": ["enter"]})
    out = layout.write_results(sd / "variations" / "baseline", _report(), decisions=dec, parameters={}, inputs=[],
                               diagnostics={"cost_model_coverage": 0.93})
    d = json.load(open(out / "diagnostics.json"))
    assert d["decisions"] == dec.to_dict(orient="records") and d["cost_model_coverage"] == 0.93


def test_the_factsheet_is_a_self_contained_page_with_the_headline_numbers(tmp_path: Path) -> None:
    sd = _set(tmp_path); r = _report()
    out = layout.write_results(sd / "variations" / "baseline", r, decisions=pd.DataFrame(), parameters={"n_hold": 8}, inputs=[sd / "bars.parquet"])
    page = _Page(); page.feed((out / "factsheet.html").read_text())
    assert "script" not in page.tags and "link" not in page.tags
    assert page.links == ["../../../report.html"], page.links        # the ONE link: up to the set report, relative
    assert not any(l.startswith(("http:", "https:", "//")) for l in page.links)
    assert {"svg", "path", "table"} <= page.tags                   # the equity curve is inline SVG
    text = " ".join(page.text)
    for key in list(r.kpis())[:3]:
        assert key in text
    assert "n_hold" in text and "bars.parquet" in text
    assert {"matrix", "tiles", "hero"} <= page.classes             # the year × month matrix and the KPI row
    assert "YTD" in text and "Total return" in text


def test_the_set_report_puts_every_variation_side_by_side_and_links_the_factsheets(tmp_path: Path) -> None:
    sd = _set(tmp_path, ("baseline", "bctrot"))
    for v, scale in (("baseline", 37.5), ("bctrot", 12.0)):
        layout.write_results(sd / "variations" / v, _report(scale), decisions=pd.DataFrame(), parameters={"variant": v}, inputs=[])
    out = layout.write_set_report(sd)
    assert out == sd / "report.html"
    page = _Page(); page.feed(out.read_text())
    assert "script" not in page.tags
    assert sorted(page.links) == ["variations/baseline/results/factsheet.html", "variations/bctrot/results/factsheet.html"]
    text = " ".join(page.text)
    assert "baseline" in text and "bctrot" in text and "total_return_pct" in text
    assert page.tags >= {"svg", "path"}
    assert "By year" in text


def test_a_stray_file_in_results_is_refused(tmp_path: Path) -> None:
    sd = _set(tmp_path); v = sd / "variations" / "baseline"
    (v / "results").mkdir(); (v / "results" / "notes.txt").write_text("x")
    with pytest.raises(RuntimeError, match="notes.txt"):
        layout.write_results(v, _report(), decisions=pd.DataFrame(), parameters={}, inputs=[])


# ---- the layout IN THE TREE ------------------------------------------------------------------

def _sets() -> list[Path]:
    return sorted(p for p in (PKG / "strategies").glob("*/backtests/*") if p.is_dir() and (p / "variations").is_dir())


def test_every_set_has_variations_with_run_and_parameters_and_no_stray_result_files() -> None:
    sets = _sets()
    assert sets, "no backtest set with variations/ in the tree — the layout test is asserting over nothing"
    for s in sets:
        variations = sorted(v for v in (s / "variations").iterdir() if v.is_dir())
        assert variations, f"{s}: variations/ is empty"
        for v in variations:
            for f in layout.VARIATION_FILES:
                assert (v / f).is_file(), f"{s.name}/{v.name}: missing {f}"
            json.loads((v / "parameters.json").read_text())
            if (v / "results").is_dir():
                names = sorted(p.name for p in (v / "results").iterdir())
                assert names == sorted(layout.RESULT_FILES), f"{s.name}/{v.name}/results: {names}"


def test_the_ledger_book_set_ships_its_data_and_bct_is_a_variation_of_momentum() -> None:
    # Falk 2026-09-17: bct is a not-yet-successful VARIATION of momentum rotation, not a strategy.
    s = PKG / "strategies" / "momentum_rotation" / "backtests" / "ledger-book-daily-2024-2026"
    assert (s / "bars.parquet").is_file() and (s / "ledger.csv").is_file()
    assert {"best", "bctrot"} <= {v.name for v in (s / "variations").iterdir() if v.is_dir()}
    p = layout.load_parameters(s / "variations" / "best")
    assert p["portfolio"]["n_hold"] == 8 and p["exits"]["give_back_frac"] == 0.5 and p["source"]["max_open_days"] == 57
    ledger = pd.read_csv(s / "ledger.csv")
    assert list(ledger.columns) == ["symbol", "open_date", "close_date", "close_pct", "still_open", "close_source"]


def test_every_set_ships_its_data_a_readme_a_report_and_one_run_py() -> None:
    """The set IS the data (Falk 2026-09-17): a set folder with no `bars.parquet` is a study, not a set.
    `run.py` is the same file in every variation — the variation is its parameters.json."""
    for s in _sets():
        assert (s / "bars.parquet").is_file(), f"{s.name}: no bars.parquet — a set carries its data"
        assert (s / "README.md").is_file(), f"{s.name}: no README"
        assert (s / "report.html").is_file(), f"{s.name}: no report.html — run a variation to generate it"
        runs = {(v / "run.py").read_bytes() for v in (s / "variations").iterdir() if v.is_dir()}
        assert len(runs) == 1, f"{s.name}: variations carry {len(runs)} different run.py files"


def test_every_variation_has_results_whose_provenance_matches_the_data_it_names() -> None:
    """A variation without results has not been run; results whose input hashes do not match the set's
    current files were produced from other data. Both are refused by name."""
    for s in _sets():
        for v in sorted(p for p in (s / "variations").iterdir() if p.is_dir()):
            prov = v / "results" / "provenance.json"
            assert prov.is_file(), f"{s.name}/{v.name}: never run"
            p = json.loads(prov.read_text())
            assert p["set"] == s.name and p["variation"] == v.name
            assert p["inputs"], f"{s.name}/{v.name}: provenance names no inputs"
            for i in p["inputs"]:
                f = s / i["path"]
                assert f.is_file(), f"{s.name}/{v.name}: provenance names {i['path']}, not in the set"
                assert layout._sha256_file(f) == i["sha256"], \
                    f"{s.name}/{v.name}: {i['path']} changed since the results were written — rerun"
            params = layout.load_parameters(v)
            assert layout._sha256_bytes(layout._canon(params)) == p["parameters_sha256"], \
                f"{s.name}/{v.name}: parameters.json changed since the results were written — rerun"
