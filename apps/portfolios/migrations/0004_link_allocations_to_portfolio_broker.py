from django.db import migrations


def link_allocations_to_broker(apps, schema_editor):
    Portfolio = apps.get_model("portfolios", "Portfolio")
    Allocation = apps.get_model("portfolios", "Allocation")
    BrokerAccount = apps.get_model("brokers", "BrokerAccount")

    for portfolio in Portfolio.objects.all():
        active_broker = BrokerAccount.objects.filter(portfolio=portfolio, trading_enabled=True).first()
        if not active_broker:
            active_broker = BrokerAccount.objects.filter(portfolio=portfolio).first()
        if active_broker:
            Allocation.objects.filter(portfolio=portfolio, broker_account__isnull=True).update(
                broker_account=active_broker
            )


def reverse_func(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("portfolios", "0003_portfolio_rebalance_frequency_and_more"),
        ("brokers", "0002_initial"),
    ]

    operations = [
        migrations.RunPython(link_allocations_to_broker, reverse_func),
    ]
