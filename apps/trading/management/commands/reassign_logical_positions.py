from decimal import Decimal

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.accounting.models import LogicalPosition
from apps.instruments.models import Instrument
from apps.portfolios.models import Portfolio
from apps.strategies.models import Strategy


class Command(BaseCommand):
    help = (
        "Corrige posiciones lógicas atribuidas a la estrategia equivocada: fusiona todas las "
        "LogicalPosition de un símbolo dentro de un portafolio en la estrategia indicada. "
        "Por defecto solo muestra lo que haría; usa --apply para escribir."
    )

    def add_arguments(self, parser):
        parser.add_argument("--portfolio", type=int, required=True, help="ID del portafolio")
        parser.add_argument("--symbol", type=str, required=True, help="Símbolo a reasignar (ej. QQQ)")
        parser.add_argument("--to-strategy", type=str, required=True, help="Slug de la estrategia dueña real de la posición")
        parser.add_argument("--apply", action="store_true", help="Aplica los cambios (sin esto es dry-run)")

    def handle(self, *args, **options):
        portfolio = Portfolio.objects.filter(id=options["portfolio"]).first()
        if not portfolio:
            raise CommandError(f"Portafolio {options['portfolio']} no encontrado.")
        instrument = Instrument.objects.filter(symbol=options["symbol"].upper()).first()
        if not instrument:
            raise CommandError(f"Instrumento {options['symbol']} no encontrado.")
        strategy = Strategy.objects.filter(slug=options["to_strategy"]).first()
        if not strategy:
            slugs = ", ".join(
                portfolio.allocations.select_related("strategy").values_list("strategy__slug", flat=True)
            )
            raise CommandError(f"Estrategia '{options['to_strategy']}' no encontrada. Slugs asignados a este portafolio: {slugs}")

        positions = list(
            LogicalPosition.objects.filter(portfolio=portfolio, instrument=instrument)
            .select_related("strategy", "broker_account")
        )
        if not positions:
            self.stdout.write("No hay posiciones lógicas para ese símbolo en el portafolio.")
            return

        self.stdout.write(f"Portafolio: {portfolio.name} | Símbolo: {instrument.symbol} | Destino: {strategy.name} ({strategy.slug})")
        by_broker = {}
        for pos in positions:
            self.stdout.write(f"  - {pos.strategy.name}: qty={pos.qty} avg_cost={pos.avg_cost} broker_account={pos.broker_account_id}")
            by_broker.setdefault(pos.broker_account_id, []).append(pos)

        with transaction.atomic():
            for broker_account_id, group in by_broker.items():
                total_qty = sum((p.qty for p in group), Decimal("0"))
                total_cost = sum((p.qty * p.avg_cost for p in group), Decimal("0"))
                total_realized = sum((p.realized_pnl for p in group), Decimal("0"))
                avg_cost = total_cost / total_qty if total_qty > 0 else Decimal("0")

                self.stdout.write(self.style.NOTICE(
                    f"  => {strategy.slug} (broker_account={broker_account_id}): qty={total_qty} avg_cost={avg_cost:.6f}"
                ))
                if not options["apply"]:
                    continue

                dest = next((p for p in group if p.strategy_id == strategy.id), None)
                if dest is None:
                    dest = LogicalPosition(
                        portfolio=portfolio,
                        broker_account_id=broker_account_id,
                        strategy=strategy,
                        instrument=instrument,
                    )
                dest.qty = total_qty
                dest.avg_cost = avg_cost
                dest.realized_pnl = total_realized
                dest.save()
                for p in group:
                    if p.pk != dest.pk:
                        p.delete()

        if options["apply"]:
            self.stdout.write(self.style.SUCCESS("Posiciones reasignadas."))
        else:
            self.stdout.write(self.style.WARNING("Dry-run: no se escribió nada. Repite con --apply para aplicar."))
