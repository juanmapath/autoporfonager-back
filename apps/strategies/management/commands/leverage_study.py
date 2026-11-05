import json
from pathlib import Path

import yaml
from django.core.management.base import BaseCommand, CommandError

from apps.strategies.leverage.report_html import render_report
from apps.strategies.leverage.simulator import SimCosts
from apps.strategies.leverage.study import LeverageStudy, StudyConfig
from apps.strategies.models import Strategy, StrategyVersion

DEFAULT_GRID = Path(__file__).resolve().parents[2] / "leverage" / "grids" / "default_grid.yaml"


def load_configs(path, version_max: float):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    out = []
    for item in raw:
        pol = dict(item.get("policy") or {})
        base = item.get("base")
        lmax = item.get("max")
        base = version_max if base == "max" else (float(base) if base is not None else None)
        lmax = version_max if lmax == "max" else (float(lmax) if lmax is not None else None)
        if base is not None and pol.get("type") == "fixed" and lmax is None:
            lmax = base
        n = sum(len(pol.get(k) or []) for k in ("scalers", "caps", "label_policies"))
        out.append(StudyConfig(str(item["name"]), str(item.get("group", "")), pol, base, lmax, n))
    return out


class Command(BaseCommand):
    help = ("Estudio de apalancamiento (04e §6): backtestea una grilla de politicas sobre las mismas señales y "
            "recomienda segun el criterio de caida maxima vs benchmark.")

    def create_parser(self, prog_name, subcommand, **kwargs):
        kwargs.setdefault("conflict_handler", "resolve")  # --version = ID de la version del bot
        return super().create_parser(prog_name, subcommand, **kwargs)

    def add_arguments(self, parser):
        parser.add_argument("--version", type=int)
        parser.add_argument("--bot", help="slug o nombre del bot (version live)")
        parser.add_argument("--configs", default=str(DEFAULT_GRID))
        parser.add_argument("--start", default="2007-01-01")
        parser.add_argument("--end", default=None)
        parser.add_argument("--split", default="2017-01-01", help="inicio de la segunda mitad (walk-forward)")
        parser.add_argument("--proxy", action="append", default=[], help="SYM=PROXY, p. ej. QQQ=^NDX")
        parser.add_argument("--history-start", default="1985-01-01")
        parser.add_argument("--margin-spread", type=float, default=0.02)
        parser.add_argument("--maintenance", type=float, default=0.30)
        parser.add_argument("--cash-yield", action="store_true")
        parser.add_argument("--report", default=None, help="ruta del HTML")
        parser.add_argument("--json", default=None, help="ruta de un JSON con la tabla")

    def handle(self, *args, **o):
        if o["version"]:
            version = StrategyVersion.objects.select_related("strategy").filter(id=o["version"]).first()
        elif o["bot"]:
            s = Strategy.objects.filter(slug=o["bot"]).first() or Strategy.objects.filter(name__iexact=o["bot"]).first()
            version = s.versions.filter(status="live").first() if s else None
        else:
            raise CommandError("Indica --version <id> o --bot <slug|nombre>.")
        if not version:
            raise CommandError("Version no encontrada.")
        proxy = {}
        for p in o["proxy"]:
            if "=" not in p:
                raise CommandError("--proxy espera SYM=PROXY")
            k, v = p.split("=", 1)
            proxy[k.strip().upper()] = v.strip()
        costs = SimCosts(margin_spread=o["margin_spread"], maintenance=o["maintenance"], cash_yield=o["cash_yield"])
        study = LeverageStudy(version, start=o["start"], end=o["end"], split=o["split"], proxy=proxy, costs=costs,
                              history_start=o["history_start"])
        configs = load_configs(o["configs"], study.max)
        out = study.run_grid(configs)
        title = f"Estudio de apalancamiento · {version.strategy.name} (v{version.version})"
        if o["report"]:
            Path(o["report"]).write_text(render_report(study, out, title), encoding="utf-8")
            self.stdout.write(f"Reporte: {o['report']}")
        rows = []
        for r in sorted(out["results"], key=lambda r: (not r.acceptable_both, -r.oos_score)):
            rows.append({
                "name": r.config.name, "acceptable": r.acceptable_both, "oos_cagr": r.oos_score,
                "cagr": r.full["cagr"], "max_dd": r.full["max_dd"], "bench_dd": r.full["bench"]["max_dd"],
                "cagr_a": r.half_a["cagr"], "cagr_b": r.half_b["cagr"], "sharpe": r.full["sharpe"],
                "avg_leverage": r.stats["avg_leverage"], "margin_calls": r.stats["margin_calls"],
                "reasons": r.full["reasons"] + r.half_a["reasons"] + r.half_b["reasons"],
            })
        if o["json"]:
            Path(o["json"]).write_text(json.dumps({"rows": rows, "winner": out["selection"]["winner"].config.name
                                                   if out["selection"]["winner"] else None,
                                                   "log": out["selection"]["log"]}, indent=2, default=str),
                                       encoding="utf-8")
        self.stdout.write(f"{'Configuracion':<52}{'Acept.':>7}{'CAGR OOS':>10}{'CAGR':>8}{'MaxDD':>8}{'Bench':>8}{'Lev':>6}")
        for r in rows:
            self.stdout.write(f"{r['name'][:51]:<52}{'si' if r['acceptable'] else 'no':>7}{r['oos_cagr'] * 100:>9.2f}%"
                              f"{r['cagr'] * 100:>7.2f}%{r['max_dd'] * 100:>7.1f}%{r['bench_dd'] * 100:>7.1f}%"
                              f"{r['avg_leverage']:>6.2f}")
        w = out["selection"]["winner"]
        self.stdout.write(f"Recomendada: {w.config.name if w else 'ninguna'}")
        for line in out["selection"]["log"]:
            self.stdout.write(f"  - {line}")
