from django.core.management.base import BaseCommand

from apps.instruments.models import Instrument
from apps.marketdata.services import MarketDataService


class Command(BaseCommand):
    help = (
        "Descarga el historial diario completo disponible de cada instrumento operable. "
        "Ejecutar una vez tras el despliegue para que el primer pipeline del día no cargue con la descarga."
    )

    def add_arguments(self, parser):
        parser.add_argument("--symbols", nargs="*", help="Limitar a estos símbolos (por defecto, todos los operables)")
        parser.add_argument("--force", action="store_true", help="Volver a descargar aunque ya esté marcado como completo")

    def handle(self, *args, **options):
        instruments = Instrument.objects.filter(tradable=True).order_by("symbol")
        if options["symbols"]:
            instruments = instruments.filter(symbol__in=[s.upper() for s in options["symbols"]])

        svc = MarketDataService()
        for inst in instruments:
            if inst.full_history_synced and not options["force"]:
                self.stdout.write(f"  {inst.symbol}: ya completo, omitido")
                continue
            if options["force"]:
                inst.full_history_synced = False
            count = svc.ensure_history(inst, incremental=False)
            first = inst.bars.order_by("date").values_list("date", flat=True).first()
            if count:
                self.stdout.write(self.style.SUCCESS(f"  {inst.symbol}: {count} barras (desde {first})"))
            else:
                self.stdout.write(self.style.WARNING(f"  {inst.symbol}: sin datos del proveedor"))
