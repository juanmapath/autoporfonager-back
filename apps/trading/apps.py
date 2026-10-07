from django.apps import AppConfig

class TradingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.trading"

    def ready(self):
        try:
            import apps.trading.tasks  # noqa: F401
        except Exception:
            pass

