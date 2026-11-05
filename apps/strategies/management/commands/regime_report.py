import json

from django.core.management.base import BaseCommand, CommandError

from apps.strategies.models import Strategy, StrategyVersion
from apps.strategies.regimes import REGIME_PROVIDERS
from apps.strategies.regimes.report import build_regime_report, format_report


class Command(BaseCommand):
    help = (
        "Informe de regimenes de una version: por regimen, trades, PF, retorno medio, apalancamiento resultante y "
        "duracion media; y backtests con politica fija (base), fija (max) y PF por regimen."
    )

    def create_parser(self, prog_name, subcommand, **kwargs):
        # `--version` es el ID de la version del bot (como pide el plan): sustituye al de Django
        kwargs.setdefault("conflict_handler", "resolve")
        return super().create_parser(prog_name, subcommand, **kwargs)

    def add_arguments(self, parser):
        parser.add_argument("--version", type=int, help="ID de StrategyVersion")
        parser.add_argument("--bot", help="Slug o nombre del bot (usa su version live)")
        parser.add_argument("--provider", default="macro", choices=sorted(REGIME_PROVIDERS))
        parser.add_argument("--start", default="1999-03-10")
        parser.add_argument("--end", default=None)
        parser.add_argument("--initial-cash", type=float, default=100000.0)
        parser.add_argument("--json", action="store_true", help="Imprime el informe como JSON")

    def handle(self, *args, **opts):
        if opts["version"]:
            version = StrategyVersion.objects.select_related("strategy").filter(id=opts["version"]).first()
        elif opts["bot"]:
            strat = Strategy.objects.filter(slug=opts["bot"]).first() or Strategy.objects.filter(name__iexact=opts["bot"]).first()
            version = strat.versions.filter(status="live").first() if strat else None
        else:
            raise CommandError("Indica --version <id> o --bot <slug|nombre>.")
        if not version:
            raise CommandError("Version no encontrada.")
        report = build_regime_report(
            version, provider=opts["provider"], start_date=opts["start"], end_date=opts["end"],
            initial_cash=opts["initial_cash"],
        )
        self.stdout.write(json.dumps(report, indent=2, default=str) if opts["json"] else format_report(report))
