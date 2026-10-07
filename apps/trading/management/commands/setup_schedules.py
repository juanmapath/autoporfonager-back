from apps.trading.management.commands.setup_botops_schedules import Command as BaseScheduleCommand

class Command(BaseScheduleCommand):
    help = "Alias for setup_botops_schedules"
