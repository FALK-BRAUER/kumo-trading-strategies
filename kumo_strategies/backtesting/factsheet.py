"""The pages a backtest set writes: a variation's factsheet and the set's report (ks#240).

SELF-CONTAINED, NO SCRIPTS. Every page is one HTML file with inline CSS and inline SVG — readable
from `file://`, from a GitHub checkout, from a mail attachment — because the evidence must be
readable where it is tracked, not where a server happens to run. The cost is no hover layer; every
number a tooltip would show is in the tables below the chart.

The visual method is the data-viz procedure (form first, colour by job, validated palette, thin
marks, recessive chrome, table view always present). Colour jobs used here, and only these:
  * EMPHASIS on the equity chart: the variation in the accent hue, benchmarks in the de-emphasis
    gray — the strategy is the point, the index is context.
  * CATEGORICAL on the set report: variations in fixed slot order (blue, orange, aqua, ...), never
    cycled; a set with more than eight variations folds the rest into the table.
  * DIVERGING on the monthly matrix: blue for gains, red for losses, a neutral gray at zero, equal
    steps per arm — polarity is the job, and the sign must never be carried by hue alone, so every
    cell also prints its number.
Light and dark are both selected (the OS setting), stepped from the same ramps.
"""
from __future__ import annotations

import html
import math
import re

import pandas as pd

# ---- palette (the reference instance of the data-viz method; swap values, keep roles) ----------
CSS = """
<style>
:root{color-scheme:light;
 --surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
 --border:rgba(11,11,11,.10);--accent:#2a78d6;--dim:#b8b7b0;--good:#006300;--bad:#d03b3b;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;--s8:#e34948;
 --pos5:#0d366b;--pos4:#1c5cab;--pos3:#3987e5;--pos2:#86b6ef;--pos1:#cde2fb;--zero:#f0efec;
 --neg1:#f7d4d4;--neg2:#eda5a5;--neg3:#e06b6b;--neg4:#c23d3d;--neg5:#7a1f1f}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;
 --surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
 --border:rgba(255,255,255,.10);--accent:#3987e5;--dim:#5a5a56;--good:#0ca30c;--bad:#e66767;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;
 --pos5:#9ec5f4;--pos4:#6da7ec;--pos3:#3987e5;--pos2:#1c5cab;--pos1:#0d366b;--zero:#383835;
 --neg1:#4a1f1f;--neg2:#7a2a2a;--neg3:#b03a3a;--neg4:#d95c5c;--neg5:#f0a0a0}}
html{background:var(--plane)}
body{font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);margin:0 auto;padding:32px 24px 64px;max-width:1080px;box-sizing:border-box}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.crumb{color:var(--muted);font-size:12px;margin:0 0 6px}
h1{font-size:26px;font-weight:600;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;font-weight:600;margin:36px 0 10px;color:var(--ink-2);text-transform:uppercase;letter-spacing:.06em}
.what{font-size:16px;color:var(--ink-2);max-width:60rem;margin:0 0 4px}
.meta{color:var(--muted);font-size:13px;margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:0 0 8px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.tile .l{font-size:12px;color:var(--muted);margin-bottom:4px}
.tile .v{font-size:24px;font-weight:600;line-height:1.1}
.tile.hero{grid-column:span 2}.tile.hero .v{font-size:38px;white-space:nowrap}
.tile .d{font-size:12px;color:var(--ink-2);margin-top:4px}.tile.hero .d{font-size:13px}
.up{color:var(--good)}.down{color:var(--bad)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px 8px;margin:0 0 12px}
.card .t{font-size:13px;color:var(--ink-2);margin:0 0 8px;display:flex;gap:18px;flex-wrap:wrap}
.card .t .k{display:inline-flex;align-items:center;gap:6px}.sw{width:14px;height:3px;border-radius:2px;display:inline-block}
svg{display:block;width:100%;height:auto}
table{border-collapse:collapse;width:100%;background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden;font-size:13px}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--grid);vertical-align:top}
tr:last-child td,tr:last-child th{border-bottom:0}
th{color:var(--ink-2);font-weight:500;background:transparent}
thead th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.matrix td{text-align:center;font-variant-numeric:tabular-nums;padding:5px 4px;min-width:44px;font-size:12px}
table.matrix td.ytd{font-weight:600;border-left:2px solid var(--axis)}
table.matrix th{white-space:nowrap}
table.kv th{width:32%;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;font-weight:400}
table.kv td{font-variant-numeric:tabular-nums;word-break:break-word}
details{margin:0 0 12px}summary{cursor:pointer;color:var(--ink-2);font-size:13px;padding:6px 0}
.foot{color:var(--muted);font-size:12px;margin-top:40px}
@media print{body{padding:0}.tile,.card,table{border-color:#ccc}}
</style>
"""

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SLOTS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"]


def _esc(x) -> str:
    return html.escape(str(x))


def _num(v, fmt: str, dash: str = "—") -> str:
    return fmt.format(v) if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else dash


def _pct(v, signed: bool = True, digits: int = 1) -> str:
    return _num(v, ("{:+." if signed else "{:.") + str(digits) + "f} %")


# ---- tiles ------------------------------------------------------------------------------------

def _tile(label: str, value: str, detail: str = "", cls: str = "", vcls: str = "") -> str:
    return (f"<div class='tile {cls}'><div class='l'>{_esc(label)}</div>"
            f"<div class='v {vcls}'>{_esc(value)}</div>" + (f"<div class='d'>{detail}</div>" if detail else "") + "</div>")


def kpi_tiles(m: dict) -> str:
    """The headline as a KPI row: the total return is the hero, the rest support it."""
    ret = m.get("total_return_pct")
    bench = [(k[len("bench_"):-len("_return_pct")], m[k]) for k in m if k.startswith("bench_") and k.endswith("_return_pct")]
    bench_line = " · ".join(f"{_esc(n)} {_pct(v)}" for n, v in bench)
    tiles = [
        _tile("Total return", _pct(ret), (f"vs {bench_line}" if bench_line else f"gross {_pct(m.get('gross_return_pct'))}"),
              "hero", "up" if isinstance(ret, (int, float)) and ret >= 0 else "down"),
        _tile("Annualised return", _pct(m.get("cagr_pct")), "CAGR", "", "up" if isinstance(m.get("cagr_pct"), (int, float)) and m["cagr_pct"] >= 0 else "down"),
        _tile("Ann. volatility", _pct(m.get("ann_vol_pct"), signed=False), f"Sortino {_num(m.get('sortino'), '{:.2f}')}"),
        _tile("Sharpe", _num(m.get("sharpe"), "{:.2f}"), f"deflated p {_num(m.get('deflated_sharpe_prob'), '{:.2f}')} · {_num(m.get('n_trials_corrected_for'), '{:,}')} trial(s)"),
        _tile("Max drawdown", _pct(m.get("max_drawdown_pct"), signed=False), f"{_num(m.get('longest_drawdown_days'), '{:,}')} sessions underwater at the longest"),
        _tile("Calmar", _num(m.get("calmar"), "{:.2f}")),
        _tile("Round trips", _num(m.get("round_trips"), "{:,}"), f"win rate {_pct(m.get('win_rate_pct'), signed=False)} · PF {_num(m.get('profit_factor'), '{:.2f}')}"),
        _tile("Avg hold", f"{_num(m.get('avg_hold_days'), '{:.1f}')} d", f"avg win {_pct(m.get('avg_win_pct'))} · avg loss {_pct(m.get('avg_loss_pct'))}"),
        _tile("Cost", _pct(m.get("cost_pct_of_capital"), signed=False), f"${_num(m.get('total_cost_usd'), '{:,.0f}')} on ${_num(m.get('turnover_usd'), '{:,.0f}')} turnover"),
    ]
    return "<div class='tiles'>" + "".join(tiles) + "</div>"


# ---- the equity chart -------------------------------------------------------------------------

def _ticks(idx: pd.DatetimeIndex) -> list[tuple[int, str]]:
    """Year-boundary and quarter ticks, thinned so labels never collide."""
    out = []
    last_label = None
    for i, d in enumerate(idx):
        key = (d.year, (d.month - 1) // 3)
        if key != last_label:
            last_label = key
            out.append((i, d.strftime("%b %Y") if d.month == 1 or not out else d.strftime("%b")))
    if len(out) > 14:
        out = [t for t in out if t[1].endswith(tuple(str(y) for y in range(2000, 2100)))]
    return out


def equity_svg(series: dict[str, pd.Series], emphasis: str | None = None, width: int = 960, height: int = 300,
               drawdown: pd.Series | None = None) -> str:
    """Growth of 1 for every series; the first (or `emphasis`) in the accent, the rest as context.
    Under it, the emphasised series' drawdown as a filled area — the underwater chart — because
    'max drawdown −23 %' says less than seeing how long it stayed there."""
    series = {k: s.dropna() for k, s in series.items() if s is not None and len(s.dropna()) > 1}
    if not series:
        return ""
    norm = {k: (s / float(s.iloc[0])) for k, s in series.items()}
    lo = min(float(n.min()) for n in norm.values())
    hi = max(float(n.max()) for n in norm.values())
    if hi <= lo:
        lo, hi = lo - 0.01, hi + 0.01
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    L, R, T = 52, 150, 14
    dd_h = 70 if drawdown is not None and len(drawdown) else 0
    plot_h = height - T - 28
    main_h = plot_h - (dd_h + 16 if dd_h else 0)
    W = width - L - R
    first = next(iter(norm.values()))
    idx = pd.to_datetime(first.index)
    n_pts = len(first)
    x = lambda i: L + i * W / max(n_pts - 1, 1)
    y = lambda v: T + (hi - v) / (hi - lo) * main_h
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="growth of 1 and drawdown" font-family="system-ui,sans-serif" font-size="11">']
    # gridlines + y labels (growth ×)
    steps = _nice_steps(lo, hi, 5)
    for v in steps:
        yy = y(v)
        out.append(f'<line x1="{L}" x2="{L + W}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--grid)" stroke-width="1"/>'
                   f'<text x="{L - 8}" y="{yy + 4:.1f}" text-anchor="end" fill="var(--muted)">{v:.2f}×</text>')
    one = y(1.0)
    if lo <= 1.0 <= hi:
        out.append(f'<line x1="{L}" x2="{L + W}" y1="{one:.1f}" y2="{one:.1f}" stroke="var(--axis)" stroke-width="1"/>')
    # x ticks
    for i, label in _ticks(idx):
        xx = x(i)
        out.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{T + main_h}" y2="{T + main_h + 4}" stroke="var(--axis)"/>'
                   f'<text x="{xx:.1f}" y="{T + main_h + 16}" text-anchor="middle" fill="var(--muted)">{_esc(label)}</text>')
    # series: context first (so the emphasis draws on top)
    keys = list(norm)
    emph = emphasis if emphasis in norm else keys[0]
    order = [k for k in keys if k != emph] + [emph]
    ends = []
    for k in order:
        s = norm[k]
        # align to the first series' index by position where lengths match, else by date
        if len(s) == n_pts:
            pts = [(x(i), y(float(v))) for i, v in enumerate(s.values)]
        else:
            si = pd.to_datetime(s.index)
            pos = {d: i for i, d in enumerate(idx)}
            pts = [(x(pos[d]), y(float(v))) for d, v in zip(si, s.values) if d in pos]
        if not pts:
            continue
        d_attr = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        stroke = "var(--accent)" if k == emph else "var(--dim)"
        sw = 2 if k == emph else 1.5
        out.append(f'<path d="{d_attr}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>')
        ends.append((pts[-1], k, stroke, float(s.iloc[-1]), k == emph))
    # end labels, nudged apart top-down so they never overlap; leader line when moved
    ends.sort(key=lambda e: e[0][1])
    last_y = -1e9
    for (ex, ey), name, stroke, end_val, is_emph in ends:
        ly = max(ey, last_y + 14); last_y = ly
        out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3" fill="{stroke}" stroke="var(--surface)" stroke-width="2"/>'
                   + (f'<line x1="{ex + 3:.1f}" y1="{ey:.1f}" x2="{ex + 8:.1f}" y2="{ly:.1f}" stroke="{stroke}"/>' if abs(ly - ey) > 2 else "")
                   + f'<text x="{ex + 10:.1f}" y="{ly + 4:.1f}" fill="{"var(--ink)" if is_emph else "var(--muted)"}" '
                   f'font-weight="{600 if is_emph else 400}">{_esc(name)} {end_val:.2f}×</text>')
    # drawdown strip
    if dd_h:
        dd = drawdown.dropna()
        top = T + main_h + 28
        ddmin = min(float(dd.min()), -0.0001)
        yd = lambda v: top + (0 - v) / (0 - ddmin) * dd_h
        if len(dd) == n_pts:
            pts = [(x(i), yd(float(v))) for i, v in enumerate(dd.values)]
        else:
            di = pd.to_datetime(dd.index)
            pos = {d: i for i, d in enumerate(idx)}
            pts = [(x(pos[d]), yd(float(v))) for d, v in zip(di, dd.values) if d in pos]
        if pts:
            area = f"M{pts[0][0]:.1f},{top:.1f} " + " ".join(f"L{px:.1f},{py:.1f}" for px, py in pts) + f" L{pts[-1][0]:.1f},{top:.1f} Z"
            out.append(f'<line x1="{L}" x2="{L + W}" y1="{top}" y2="{top}" stroke="var(--axis)"/>'
                       f'<path d="{area}" fill="var(--bad)" fill-opacity=".18" stroke="var(--bad)" stroke-width="1"/>'
                       f'<text x="{L - 8}" y="{top + 4}" text-anchor="end" fill="var(--muted)">0 %</text>'
                       f'<text x="{L - 8}" y="{top + dd_h + 4}" text-anchor="end" fill="var(--muted)">{ddmin * 100:.0f} %</text>'
                       f'<text x="{L + W + 8}" y="{top + 12}" fill="var(--muted)">drawdown</text>')
    out.append("</svg>")
    return "".join(out)


def _nice_steps(lo: float, hi: float, n: int) -> list[float]:
    span = hi - lo
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n + 1:
            break
    start = math.ceil(lo / step) * step
    return [round(start + i * step, 6) for i in range(int((hi - start) / step) + 1)]


# ---- the monthly matrix -----------------------------------------------------------------------

def monthly_matrix(monthly: pd.DataFrame | pd.Series, equity: pd.Series | None = None) -> str:
    """Year × month returns, coloured on the diverging scale with the number printed in every cell,
    a YTD column per year (compounded from the months, so it is the year's own return), and a
    'Year' row that sums nothing — the whole point of a matrix is that no cell is an average."""
    s = monthly["return_pct"] if isinstance(monthly, pd.DataFrame) else monthly
    s = s.dropna()
    if s.empty:
        return ""
    idx = pd.to_datetime(s.index)
    years = sorted({d.year for d in idx})
    by = {(d.year, d.month): float(v) for d, v in zip(idx, s.values)}
    scale = max(5.0, min(30.0, float(abs(s).quantile(0.95))))   # the arm's full step; clipped so one month cannot bleach the rest
    head = "<tr><th>year</th>" + "".join(f"<th class='n'>{m}</th>" for m in MONTHS) + "<th class='n ytd'>YTD</th></tr>"
    rows = []
    for yr in years:
        cells = []
        growth = 1.0
        for mo in range(1, 13):
            v = by.get((yr, mo))
            if v is None:
                cells.append("<td></td>")
                continue
            growth *= 1 + v / 100
            cells.append(f"<td style='background:{_diverging(v, scale)};color:{_ink_on(v, scale)}' title='{yr}-{mo:02d}'>{v:+.1f}</td>")
        ytd = (growth - 1) * 100
        cells.append(f"<td class='ytd' style='background:{_diverging(ytd, scale * 3)};color:{_ink_on(ytd, scale * 3)}'>{ytd:+.1f}</td>")
        rows.append(f"<tr><th>{yr}</th>{''.join(cells)}</tr>")
    legend = (f"<p class='meta' style='margin-top:8px'>monthly return, %. Colour steps every {scale / 5:.1f} pp to ±{scale:.0f} % "
              f"(YTD column on a 3× wider scale); the sign is printed, never carried by colour alone.</p>")
    return f"<table class='matrix'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>{legend}"


def _diverging(v: float, scale: float) -> str:
    """Five steps per arm from the palette's blue/red ramps, neutral gray at zero."""
    if v is None or math.isnan(v):
        return "transparent"
    k = min(5, int(math.ceil(abs(v) / scale * 5))) if abs(v) > 1e-9 else 0
    if k == 0:
        return "var(--zero)"
    return f"var(--pos{k})" if v > 0 else f"var(--neg{k})"


def _ink_on(v: float, scale: float) -> str:
    """Ink inside a coloured fill: white on the three darkest steps, page ink otherwise."""
    if v is None or math.isnan(v):
        return "var(--ink)"
    k = min(5, int(math.ceil(abs(v) / scale * 5))) if abs(v) > 1e-9 else 0
    return "#fff" if k >= 4 else "var(--ink)"


# ---- tables -----------------------------------------------------------------------------------

def kv_table(d: dict, cls: str = "kv") -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}"
        if isinstance(v, (dict, list)):
            import json
            return json.dumps(v, default=str)
        return str(v)
    return f"<table class='{cls}'>" + "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(fmt(v))}</td></tr>" for k, v in d.items()) + "</table>"


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def trades_table(trades: pd.DataFrame | None, n: int = 10) -> str:
    """The best and worst round trips — the tails a reader asks about first."""
    if trades is None or trades.empty or "return_pct" not in trades.columns:
        return ""
    want = [("symbol", "symbol"), ("side", "side"), ("entry_ts", "entry"), ("exit_ts", "exit"),
            ("hold_days", "held"), ("return_pct", "return"), ("net_pnl", "net P&L")]
    cols = [(c, label) for c, label in want if c in trades.columns]
    t = trades.sort_values("return_pct")
    worst, best = t.head(n), t.tail(n).iloc[::-1]

    def rows(df):
        out = []
        for _, r in df.iterrows():
            tds = []
            for c, _label in cols:
                v = r[c]
                if c == "return_pct":
                    tds.append(f"<td class='n {'up' if v >= 0 else 'down'}'>{v:+.1f} %</td>")
                elif c == "net_pnl":
                    tds.append(f"<td class='n'>{v:+,.0f}</td>")
                elif c in ("entry_ts", "exit_ts"):
                    tds.append(f"<td>{_esc(str(v)[:10])}</td>")
                elif c == "hold_days":
                    tds.append(f"<td class='n'>{int(v)} d</td>")
                else:
                    tds.append(f"<td>{_esc(v)}</td>")
            out.append("<tr>" + "".join(tds) + "</tr>")
        return "".join(out)
    head = "<tr>" + "".join(f"<th class='{'n' if c in ('hold_days', 'return_pct', 'net_pnl') else ''}'>{_esc(label)}</th>" for c, label in cols) + "</tr>"
    return (f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px'>"
            f"<div><p class='meta' style='margin-bottom:6px'>best {n}</p><table><thead>{head}</thead><tbody>{rows(best)}</tbody></table></div>"
            f"<div><p class='meta' style='margin-bottom:6px'>worst {n}</p><table><thead>{head}</thead><tbody>{rows(worst)}</tbody></table></div></div>")


# ---- pages ------------------------------------------------------------------------------------

def _what(name: str, text: str) -> str:
    """parameters.json's `variant` line starts with the variation's own name and a dash; the page
    already prints the name, so the sentence starts at what it IS."""
    return re.sub(r"^\s*" + re.escape(name) + r"\s*[—–-]+\s*", "", text or "")


def _window(equity: pd.Series) -> str:
    if equity is None or len(equity) == 0:
        return "no sessions"
    idx = pd.to_datetime(equity.index)
    return f"{idx[0].date()} → {idx[-1].date()} · {len(idx)} sessions"


def _drawdown(equity: pd.Series) -> pd.Series:
    e = equity.dropna()
    return e / e.cummax() - 1.0


def factsheet_page(metrics: dict, monthly, equity: pd.Series, parameters: dict, prov: dict, *,
                   benchmarks: dict[str, pd.Series] | None = None, trades: pd.DataFrame | None = None) -> str:
    strategy, sset, variation = prov.get("strategy", ""), prov.get("set", ""), prov.get("variation", "")
    what = _what(variation, parameters.get("variant", ""))
    series = {variation or "equity": equity}
    start = pd.to_datetime(equity.index).min() if len(equity) else None
    for name, b in (benchmarks or {}).items():
        # the benchmark over the SAME window as the strategy's active equity — a benchmark that starts
        # a year earlier is a year of return the strategy was never in the market to earn
        b = b.dropna()
        series[name] = b[pd.to_datetime(b.index) >= start] if start is not None else b
    inputs = {i["path"]: f"{i['bytes']:,} bytes · sha256 {i['sha256'][:16]}…" for i in prov.get("inputs", [])}
    params = {k: v for k, v in parameters.items() if k != "variant"}
    legend = "".join(f"<span class='k'><i class='sw' style='background:{'var(--accent)' if k == (variation or 'equity') else 'var(--dim)'}'></i>{_esc(k)}</span>"
                     for k in series)
    return (
        f"<!doctype html>\n<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(strategy)} · {_esc(sset)} · {_esc(variation)}</title>{CSS}</head><body>"
        f"<p class='crumb'>{_esc(strategy)} / backtests / {_esc(sset)} / variations / {_esc(variation)} · "
        f"<a href='../../../report.html'>set report</a></p>"
        f"<h1>{_esc(strategy)} · {_esc(variation)}</h1>"
        f"<p class='what'>{_esc(what)}</p>"
        f"<p class='meta'>{_esc(sset)} · {_esc(_window(equity))} · generated {_esc(prov.get('generated_at', ''))} · tree {_esc(prov.get('git_sha', '')[:12])}</p>"
        f"{kpi_tiles(metrics)}"
        f"<div class='card'><p class='t'><span>growth of 1 · </span>{legend}</p>"
        f"{equity_svg(series, emphasis=variation or 'equity', drawdown=_drawdown(equity))}</div>"
        f"<h2>Monthly returns</h2>{monthly_matrix(monthly, equity)}"
        + (f"<h2>Round trips</h2>{trades_table(trades)}" if trades is not None and not trades.empty else "")
        + f"<h2>Every metric</h2>{kv_table(metrics)}"
        f"<h2>Parameters</h2>{kv_table(_flat(params))}"
        f"<h2>Inputs</h2>{kv_table(inputs)}"
        f"<details><summary>Provenance</summary>{kv_table({k: v for k, v in prov.items() if k not in ('argv', 'inputs')})}</details>"
        f"<p class='foot'>Self-contained page: no scripts, no external resources. Every number here is reproducible from "
        f"<code>run.py</code> beside this folder; the inputs are hashed above.</p>"
        "</body></html>\n"
    )


def set_report_page(strategy: str, set_name: str, rows: dict[str, dict], curves: dict[str, pd.Series],
                    whats: dict[str, str], generated_at: str) -> str:
    first = next(iter(curves.values()), pd.Series(dtype=float))
    names = list(rows)
    legend = "".join(f"<span class='k'><i class='sw' style='background:var({SLOTS[i % 8]})'></i>{_esc(n)}</span>"
                     for i, n in enumerate(names[:8]))

    def cell(n, k, fmt, signed=True):
        v = rows[n].get(k)
        txt = _num(v, fmt)
        cls = ""
        if k == "total_return_pct" and isinstance(v, (int, float)):
            cls = "up" if v >= 0 else "down"
        return f"<td class='n {cls}'>{txt}</td>"
    summary = "".join(
        f"<tr><th><span class='k'><i class='sw' style='background:var({SLOTS[i % 8]})'></i>"
        f"<a href='variations/{_esc(n)}/results/factsheet.html'>{_esc(n)}</a></span></th>"
        f"<td>{_esc(_what(n, whats.get(n, '')))}</td>"
        + cell(n, "total_return_pct", "{:+.1f} %") + cell(n, "cagr_pct", "{:.1f} %") + cell(n, "sharpe", "{:.2f}")
        + cell(n, "max_drawdown_pct", "{:.1f} %") + cell(n, "calmar", "{:.2f}") + cell(n, "round_trips", "{:,}")
        + cell(n, "deflated_sharpe_prob", "{:.2f}") + "</tr>"
        for i, n in enumerate(names))
    keys = sorted({k for m in rows.values() for k in m})
    head = "<tr><th>metric</th>" + "".join(f"<th class='n'>{_esc(n)}</th>" for n in names) + "</tr>"
    body = "".join(
        f"<tr><th>{_esc(k)}</th>" + "".join(
            f"<td class='n'>{_esc(f'{rows[n][k]:,.4g}' if isinstance(rows[n].get(k), (int, float)) else str(rows[n].get(k, '')))}</td>"
            for n in names) + "</tr>" for k in keys)
    # yearly returns per variation, from each equity curve
    yearly_rows = ""
    years = sorted({d.year for c in curves.values() for d in pd.to_datetime(c.index)})
    if years:
        yearly_rows = "<tr><th>year</th>" + "".join(f"<th class='n'>{_esc(n)}</th>" for n in names) + "</tr>"
        for yr in years:
            tds = []
            for n in names:
                c = curves.get(n)
                if c is None or c.empty:
                    tds.append("<td></td>"); continue
                ci = pd.to_datetime(c.index)
                seg = c[(ci.year == yr)]
                if seg.empty:
                    tds.append("<td></td>"); continue
                prev = c[ci < pd.Timestamp(f"{yr}-01-01")]
                base = float(prev.iloc[-1]) if len(prev) else float(seg.iloc[0])
                v = (float(seg.iloc[-1]) / base - 1) * 100
                tds.append(f"<td class='n {'up' if v >= 0 else 'down'}'>{v:+.1f} %</td>")
            yearly_rows += f"<tr><th>{yr}</th>{''.join(tds)}</tr>"
    return (
        f"<!doctype html>\n<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(strategy)} · {_esc(set_name)}</title>{CSS}</head><body>"
        f"<p class='crumb'>{_esc(strategy)} / backtests / {_esc(set_name)}</p>"
        f"<h1>{_esc(strategy)} · {_esc(set_name)}</h1>"
        f"<p class='meta'>{_esc(_window(first))} · {len(rows)} variation(s) · generated {_esc(generated_at)} · "
        f"the set's README says what the data is and how it was scoped</p>"
        "<table><thead><tr><th>variation</th><th>what</th><th class='n'>total return</th><th class='n'>CAGR</th>"
        "<th class='n'>Sharpe</th><th class='n'>max DD</th><th class='n'>Calmar</th><th class='n'>round trips</th>"
        f"<th class='n'>deflated p</th></tr></thead><tbody>{summary}</tbody></table>"
        f"<div class='card' style='margin-top:12px'><p class='t'><span>growth of 1 · </span>{legend}</p>{_multi_svg(curves)}</div>"
        + (f"<h2>By year</h2><table><thead>{yearly_rows.split('</tr>', 1)[0]}</tr></thead><tbody>{yearly_rows.split('</tr>', 1)[1]}</tbody></table>" if yearly_rows else "")
        + f"<h2>Every metric</h2><table><thead>{head}</thead><tbody>{body}</tbody></table>"
        "<p class='foot'>Self-contained page: no scripts, no external resources. Each variation's factsheet holds its monthly "
        "matrix, round trips, parameters and provenance.</p>"
        "</body></html>\n"
    )


def _multi_svg(curves: dict[str, pd.Series], width: int = 960, height: int = 300) -> str:
    """Categorical overlay: every variation in its fixed slot colour, direct end labels, one legend."""
    curves = {k: s.dropna() for k, s in curves.items() if s is not None and len(s.dropna()) > 1}
    if not curves:
        return ""
    norm = {k: s / float(s.iloc[0]) for k, s in curves.items()}
    lo = min(float(n.min()) for n in norm.values()); hi = max(float(n.max()) for n in norm.values())
    if hi <= lo:
        lo, hi = lo - 0.01, hi + 0.01
    pad = (hi - lo) * 0.06; lo, hi = lo - pad, hi + pad
    L, R, T, B = 52, 150, 14, 28
    W, H = width - L - R, height - T - B
    longest = max(norm.values(), key=len)
    idx = pd.to_datetime(longest.index); pos = {d: i for i, d in enumerate(idx)}; n_pts = len(idx)
    x = lambda i: L + i * W / max(n_pts - 1, 1)
    y = lambda v: T + (hi - v) / (hi - lo) * H
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="growth of 1 per variation" font-family="system-ui,sans-serif" font-size="11">']
    for v in _nice_steps(lo, hi, 5):
        out.append(f'<line x1="{L}" x2="{L + W}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end" fill="var(--muted)">{v:.2f}×</text>')
    if lo <= 1 <= hi:
        out.append(f'<line x1="{L}" x2="{L + W}" y1="{y(1):.1f}" y2="{y(1):.1f}" stroke="var(--axis)"/>')
    for i, label in _ticks(idx):
        out.append(f'<text x="{x(i):.1f}" y="{T + H + 16}" text-anchor="middle" fill="var(--muted)">{_esc(label)}</text>')
    ends = []
    for k, (name, s) in enumerate(norm.items()):
        si = pd.to_datetime(s.index)
        pts = [(x(pos[d]), y(float(v))) for d, v in zip(si, s.values) if d in pos]
        if not pts:
            continue
        col = f"var({SLOTS[k % 8]})"
        out.append(f'<path d="M{" L".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        ends.append((pts[-1], name, col, float(s.iloc[-1])))
    # end labels, nudged apart so they never overlap
    ends.sort(key=lambda e: e[0][1])
    last_y = -1e9
    for (ex, ey), name, col, val in ends:
        ly = max(ey, last_y + 14); last_y = ly
        out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3" fill="{col}" stroke="var(--surface)" stroke-width="2"/>'
                   + (f'<line x1="{ex + 3:.1f}" y1="{ey:.1f}" x2="{ex + 8:.1f}" y2="{ly:.1f}" stroke="{col}" stroke-width="1"/>' if abs(ly - ey) > 2 else "")
                   + f'<text x="{ex + 10:.1f}" y="{ly + 4:.1f}" fill="var(--ink)">{_esc(name)} {val:.2f}×</text>')
    out.append("</svg>")
    return "".join(out)
