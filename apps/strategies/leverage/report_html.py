"""Reporte HTML autocontenido del estudio de apalancamiento (tablas + gráficos SVG en línea)."""
import html
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PALETTE = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626", "#0891b2"]
BENCH_COLOR = "#6b7280"


def _pct(v, nd=1):
    return "–" if v is None else f"{v * 100:.{nd}f}%"


def _num(v, nd=2):
    return "–" if v is None else f"{v:.{nd}f}"


def _svg_lines(series: List[Dict[str, Any]], width=900, height=300, log=False, pct_axis=False, title="") -> str:
    """series: [{label, values: pd.Series, color, dash}]"""
    pad_l, pad_r, pad_t, pad_b = 60, 12, 26, 28
    allv = pd.concat([s["values"] for s in series])
    allv = allv.replace([np.inf, -np.inf], np.nan).dropna()
    if allv.empty:
        return ""
    tr = (lambda x: np.log(np.maximum(x, 1e-9))) if log else (lambda x: x)
    lo, hi = float(tr(allv.min())), float(tr(allv.max()))
    if hi - lo < 1e-9:
        hi = lo + 1
    t0 = min(s["values"].index[0] for s in series)
    t1 = max(s["values"].index[-1] for s in series)
    span = max((t1 - t0).days, 1)

    def xy(d, v):
        x = pad_l + (d - t0).days / span * (width - pad_l - pad_r)
        y = pad_t + (1 - (float(tr(v)) - lo) / (hi - lo)) * (height - pad_t - pad_b)
        return x, y

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{pad_l}" y="16" class="ct">{html.escape(title)}</text>']
    for k in range(5):
        val = lo + (hi - lo) * k / 4
        y = pad_t + (1 - k / 4) * (height - pad_t - pad_b)
        real = float(np.exp(val)) if log else val
        lab = f"{real * 100:.0f}%" if pct_axis else (f"{real:.1f}x" if log else f"{real:.2f}")
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="ax" text-anchor="end">{lab}</text>')
    for yr in range(t0.year + 1, t1.year + 1, max(1, (t1.year - t0.year) // 10)):
        x, _ = xy(pd.Timestamp(f"{yr}-01-01"), allv.iloc[0])
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" class="ax" text-anchor="middle">{yr}</text>')
    for s in series:
        v = s["values"].dropna()
        step = max(1, len(v) // 700)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(d, val) for d, val in list(v.items())[::step]))
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        parts.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="1.6"{dash} points="{pts}"/>')
    legend_x = pad_l + 8
    for i, s in enumerate(series):
        y = pad_t + 14 + i * 15
        parts.append(f'<rect x="{legend_x}" y="{y - 9}" width="10" height="3" fill="{s["color"]}"/>'
                     f'<text x="{legend_x + 14}" y="{y - 4}" class="lg">{html.escape(s["label"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_report(study, out: Dict[str, Any], title: str) -> str:
    results = out["results"]
    sel = out["selection"]
    winner = sel.get("winner")
    bench = study.bench
    rows = sorted(results, key=lambda r: (not r.acceptable_both, -r.score, -r.oos_calmar))

    def crisis_cells(r):
        cells = []
        for w in study.crises:
            c = next((c for c in r.full["crises"] if c["name"] == w["name"]), None)
            if c is None:
                cells.append("<td>–</td>")
            else:
                cls = "ok" if c["ok"] else "bad"
                cells.append(f'<td class="{cls}">{_pct(c["dd"])}</td>')
        return "".join(cells)

    crisis_head = "".join(f"<th>{html.escape(w['name'])}<br><small>bench {_pct(_bench_dd(study, w))}</small></th>"
                          for w in study.crises)
    body = []
    for r in rows:
        f, a, b, st = r.full, r.half_a, r.half_b, r.stats
        verdict = "Aceptable" if r.acceptable_both else "No aceptable"
        reasons = "; ".join(dict.fromkeys(f["reasons"] + [f"2007–{study.split.year - 1}: " + x for x in a["reasons"]]
                                          + [f"{study.split.year}–: " + x for x in b["reasons"]]))
        win = " class=\"win\"" if winner is r else ""
        body.append(
            f"<tr{win}><td>{html.escape(r.config.name)}<br><small>{html.escape(r.config.group)}"
            f"{' · ' + html.escape(r.config.note) if r.config.note else ''}</small></td>"
            f"<td class=\"{'ok' if r.acceptable_both else 'bad'}\">{verdict}</td>"
            f"<td><b>{_num(r.score) if r.acceptable_both else '–'}</b></td>"
            f"<td><b>{_num(r.oos_calmar)}</b></td><td><b>{_pct(r.oos_cagr)}</b></td>"
            f"<td>{_pct(f['cagr'])}</td><td>{_pct(a['cagr'])}</td><td>{_pct(b['cagr'])}</td>"
            f"<td>{_pct(f['max_dd'])}</td><td>{_num(f['calmar'])}</td><td>{_num(f['sortino'])}</td>"
            f"<td>{_num(f['sharpe'])}</td>{crisis_cells(r)}<td>{_recovery(f['recovery_days'])}</td>"
            f"<td>{_num(st['avg_leverage'])}x</td><td>{st['pct_days_over_1x']:.0f}%</td>"
            f"<td>{_num(st['turnover_per_year'], 1)}</td><td>{st['interest_paid_pct']:.1f}%</td>"
            f"<td>{st['margin_calls']}</td><td><small>{html.escape(reasons)}</small></td></tr>"
        )
    top = [r for r in rows if r.acceptable_both][:5]
    eq_series = [{"label": f"Benchmark {study.symbol} 1x", "values": bench, "color": BENCH_COLOR, "dash": True}]
    dd_series = [{"label": f"Benchmark {study.symbol} 1x", "values": bench / bench.cummax() - 1, "color": BENCH_COLOR,
                  "dash": True}]
    for i, r in enumerate(top):
        eq_series.append({"label": r.config.name, "values": r.equity / r.equity.iloc[0], "color": PALETTE[i]})
        dd_series.append({"label": r.config.name, "values": r.equity / r.equity.cummax() - 1, "color": PALETTE[i]})
    charts = _svg_lines(eq_series, log=True, title="Equity (escala log) · 5 mejores aceptables vs benchmark")
    charts += _svg_lines(dd_series, pct_axis=True, title="Caídas desde máximos")
    if winner is not None:
        lev = winner.applied.where(winner.applied > 0)
        charts += _svg_lines([{"label": winner.config.name, "values": lev.ffill().fillna(0), "color": PALETTE[0]}],
                             height=220, title=f"Apalancamiento aplicado en el tiempo · {winner.config.name} (0 = fuera)")
    rec = _recommendation(study, out)
    rob = sel.get("robustness") or {}
    rob_rows = "".join(
        f"<tr><td>{html.escape(x['variant'])}</td><td>{_pct(x['oos_cagr'])}</td><td>{x['drop'] * 100:+.2f} pts</td>"
        f"<td class=\"{'ok' if x['ok'] else 'bad'}\">{'ok' if x['ok'] else 'falla'}</td></tr>"
        for x in rob.get("rows", []))
    sm_rows = "".join(
        f"<tr><td>{x['symbol']}</td><td>{_pct(x.get('cagr'))}</td><td>{_pct(x.get('bench_cagr'))}</td>"
        f"<td>{_pct(x.get('max_dd'))}</td><td>{_pct(x.get('bench_dd'))}</td><td>{_num(x.get('calmar'))}</td>"
        f"<td class=\"{'ok' if x['ok'] else 'bad'}\">{'ok' if x['ok'] else 'no cumple'}</td><td><small>{html.escape('; '.join(x.get('reasons', [])) or x.get('error', ''))}</small></td></tr>"
        for x in (sel.get("second_markets") or []))
    proxy = (f"<p class=\"note\">Historia extendida: antes de {study.proxy_until.date()} se usan los retornos de "
             f"{html.escape(', '.join(study.proxy.values()))} como proxy de {study.symbol}.</p>") if study.proxy_until else ""
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#fff;--fg:#0f172a;--muted:#64748b;--line:#e2e8f0;--ok:#15803d;--okbg:#f0fdf4;--bad:#b91c1c;--badbg:#fef2f2;--win:#eff6ff}}
@media (prefers-color-scheme: dark){{:root{{--bg:#0b1220;--fg:#e2e8f0;--muted:#94a3b8;--line:#1e293b;--ok:#4ade80;--okbg:#052e16;--bad:#f87171;--badbg:#450a0a;--win:#172554}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px 16px;max-width:1400px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} .note{{color:var(--muted)}}
.wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top;white-space:nowrap}}
th:first-child,td:first-child,td:last-child{{text-align:left;white-space:normal}} th{{position:sticky;top:0;background:var(--bg)}}
td.ok{{color:var(--ok);background:var(--okbg)}} td.bad{{color:var(--bad);background:var(--badbg)}} tr.win td{{background:var(--win)}}
.chart{{width:100%;height:auto;margin:8px 0;border:1px solid var(--line);border-radius:8px}}
.chart .grid{{stroke:var(--line)}} .chart .ax,.chart .lg{{fill:var(--muted);font-size:11px}} .chart .ct{{fill:var(--fg);font-size:13px;font-weight:600}}
.rec{{border:1px solid var(--line);border-left:4px solid #2563eb;border-radius:8px;padding:12px 16px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="note">{study.symbol} · {study.start.date()} → {study.end.date()} · mitades {study.start.year}–{study.split.year - 1} / {study.split.year}–{study.end.year}
· apalancamiento de la versión: base {study.base:g} / máx {study.max:g} · margen ^IRX + {study.costs.margin_spread * 100:.1f}% · mantenimiento {study.costs.maintenance:.0%}
· {out['n_configs']} configuraciones de la grilla, {out['n_runs']} simulaciones en total (incluye robustez y segundo mercado).</p>
{proxy}
<h2>Recomendación</h2><div class="rec">{rec}</div>
<h2>Gráficos</h2>{charts}
<h2>Todas las configuraciones</h2>
<p class="note">Rechazo: CAGR menor que el del benchmark ({study.symbol} 1x, CAGR {_pct(_bench_cagr(study))}) en el periodo completo o en alguna mitad, o llamadas de margen. Ranking de las aceptables: puntaje = {study.calmar_weight:.0%} × percentil del Calmar fuera de muestra + {1 - study.calmar_weight:.0%} × percentil del CAGR fuera de muestra (fuera de muestra = promedio de las dos mitades, cada una valida la selección hecha en la otra). Las caídas por crisis (verde = cae menos que el benchmark), Sortino y Sharpe son informativos.</p>
<div class="wrap"><table><thead><tr><th>Configuración</th><th>Veredicto</th><th>Puntaje</th><th>Calmar fuera de muestra</th><th>CAGR fuera de muestra</th><th>CAGR total</th><th>CAGR 1ª mitad</th><th>CAGR 2ª mitad</th><th>Caída máx.<br><small>bench {_pct(_bench_maxdd(study))}</small></th><th>Calmar total<br><small>bench {_num(_bench_calmar(study))}</small></th><th>Sortino</th><th>Sharpe</th>{crisis_head}<th>Recuperación</th><th>Apalanc. medio</th><th>% días &gt;1x</th><th>Rotación/año</th><th>Interés margen</th><th>Llamadas margen</th><th>Motivo</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<h2>Robustez del ganador (±20% por parámetro)</h2>
<div class="wrap"><table><thead><tr><th>Variante</th><th>CAGR fuera de muestra</th><th>Cambio</th><th>Resultado</th></tr></thead><tbody>{rob_rows or '<tr><td colspan=4>Sin parámetros numéricos</td></tr>'}</tbody></table></div>
<h2>Segundo mercado</h2>
<div class="wrap"><table><thead><tr><th>Mercado</th><th>CAGR</th><th>CAGR benchmark</th><th>Caída máx.</th><th>Caída benchmark</th><th>Calmar</th><th>Criterio</th><th>Detalle</th></tr></thead><tbody>{sm_rows}</tbody></table></div>
</body></html>"""


def _bench_cagr(study):
    from apps.strategies.leverage.metrics import cagr
    return cagr(study.bench)


def _bench_calmar(study):
    dd = _bench_maxdd(study)
    return _bench_cagr(study) / abs(dd) if dd < 0 else None


def _bench_maxdd(study):
    b = study.bench
    return float((b / b.cummax() - 1).min())


def _bench_dd(study, w):
    from apps.strategies.leverage.metrics import window_drawdown
    return window_drawdown(study.bench, w["start"], w["end"])


def _recovery(days):
    if days is None:
        return "–"
    if days < 0:
        return f"sin recuperar ({-days} d)"
    return f"{days} d"


def _recommendation(study, out) -> str:
    sel = out["selection"]
    w = sel.get("winner")
    log = "".join(f"<li>{html.escape(x)}</li>" for x in sel.get("log", []))
    pa, pb = sel.get("pick_a"), sel.get("pick_b")
    wf = ""
    if pa:
        wf += (f"<li>Mejor puntaje en {study.start.year}–{study.split.year - 1}: <b>{html.escape(pa.config.name)}</b> "
               f"(Calmar {_num(pa.half_a['calmar'])}, CAGR {_pct(pa.half_a['cagr'])}); en la validación "
               f"{study.split.year}–: Calmar {_num(pa.half_b['calmar'])}, CAGR {_pct(pa.half_b['cagr'])}, "
               f"{'aceptable' if pa.half_b['acceptable'] else 'NO aceptable'}.</li>")
    if pb:
        wf += (f"<li>Mejor puntaje en {study.split.year}–: <b>{html.escape(pb.config.name)}</b> "
               f"(Calmar {_num(pb.half_b['calmar'])}, CAGR {_pct(pb.half_b['cagr'])}); en la validación "
               f"{study.start.year}–{study.split.year - 1}: Calmar {_num(pb.half_a['calmar'])}, CAGR {_pct(pb.half_a['cagr'])}, "
               f"{'aceptable' if pb.half_a['acceptable'] else 'NO aceptable'}.</li>")
    if w is None:
        return f"<p><b>Ninguna configuración cumple el criterio en ambas mitades.</b></p><ul>{wf}{log}</ul>"
    ref = next((r for r in out["results"] if r.config.name.startswith("regime_pf")), None)
    vs = ""
    if ref is not None and ref is not w:
        vs = (f" Frente a la política actual (regime_pf macro): Calmar fuera de muestra {_num(ref.oos_calmar)}, "
              f"CAGR {_pct(ref.oos_cagr)}, caída máx. {_pct(ref.full['max_dd'])}, "
              f"{'aceptable' if ref.acceptable_both else 'no aceptable'}.")
    return (f"<p><b>Recomendada: {html.escape(w.config.name)}</b> — Calmar fuera de muestra {_num(w.oos_calmar)}, "
            f"CAGR fuera de muestra {_pct(w.oos_cagr)} (total {_pct(w.full['cagr'])} vs benchmark "
            f"{_pct(w.full['bench']['cagr'])}), caída máx. {_pct(w.full['max_dd'])} vs {_pct(w.full['bench']['max_dd'])}, "
            f"Sortino {_num(w.full['sortino'])}, apalancamiento medio {w.stats['avg_leverage']:.2f}x, "
            f"{w.config.n_layers} capa(s).{vs}</p><ul>{wf}{log}</ul>"
            f"<p class=\"note\">El estudio recomienda; aplicar la configuración es decisión del operador.</p>")
