from rest_framework import serializers
from apps.brokers.models import BrokerAccount, BrokerCredential
from apps.brokers.providers.alpaca import test_alpaca_credentials


class BrokerAccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = BrokerAccount
        fields = [
            "id", "portfolio", "provider", "environment",
            "execution_mode", "display_name", "broker_label",
            "external_account_id", "managed_capital", "order_policy", "trading_enabled",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class BrokerTestConnectionSerializer(serializers.Serializer):
    provider = serializers.ChoiceField(choices=["alpaca", "manual"], default="alpaca")
    environment = serializers.ChoiceField(choices=["paper", "live"], default="paper")
    api_key = serializers.CharField(required=False, allow_blank=True)
    secret_key = serializers.CharField(required=False, allow_blank=True)

    def validate(self, data):
        provider = data.get("provider")
        env = data.get("environment", "paper")
        api_key = data.get("api_key", "").strip()
        secret_key = data.get("secret_key", "").strip()

        if provider == "alpaca":
            if not api_key or not secret_key:
                raise serializers.ValidationError({"api_key": "API Key ID y Secret Key son obligatorios para Alpaca."})
            
            ping_res = test_alpaca_credentials(api_key, secret_key, env)
            if not ping_res.get("success"):
                raise serializers.ValidationError({"api_key": ping_res.get("error", "Error al conectar con Alpaca.")})
            
            data["alpaca_account"] = ping_res

        return data


class BrokerAccountCreateSerializer(serializers.Serializer):
    display_name = serializers.CharField(max_length=128)
    provider = serializers.ChoiceField(choices=BrokerAccount.PROVIDER_CHOICES, default="alpaca")
    environment = serializers.ChoiceField(choices=BrokerAccount.ENV_CHOICES, default="paper")
    broker_label = serializers.CharField(max_length=64, default="Alpaca")
    execution_mode = serializers.ChoiceField(choices=BrokerAccount.EXECUTION_MODE_CHOICES, default="auto")
    api_key = serializers.CharField(required=False, allow_blank=True)
    secret_key = serializers.CharField(required=False, allow_blank=True)

    def validate(self, data):
        provider = data.get("provider")
        env = data.get("environment", "paper")
        api_key = data.get("api_key", "").strip()
        secret_key = data.get("secret_key", "").strip()

        if provider == "alpaca":
            if not api_key or not secret_key:
                raise serializers.ValidationError({"api_key": "API Key ID y Secret Key son obligatorios para Alpaca."})

            ping_res = test_alpaca_credentials(api_key, secret_key, env)
            if not ping_res.get("success"):
                raise serializers.ValidationError({"api_key": ping_res.get("error", "Error de conexión con Alpaca.")})

            data["alpaca_account"] = ping_res

        return data

    def create_for_portfolio(self, portfolio, validated_data):
        user = portfolio.user
        provider = validated_data.get("provider", "alpaca")
        env = validated_data.get("environment", "paper")
        api_key = validated_data.get("api_key", "").strip()
        secret_key = validated_data.get("secret_key", "").strip()
        alpaca_account = validated_data.get("alpaca_account", {})

        credential = None
        if api_key and secret_key:
            credential = BrokerCredential.objects.create(
                user=user,
                provider=provider,
                environment=env,
                auth_type="api_key",
            )
            credential.set_secrets({
                "key_id": api_key,
                "secret_key": secret_key,
                "environment": env,
            })
            credential.save()

        external_id = alpaca_account.get("account_id") or alpaca_account.get("account_number") or ""
        portfolio_value = alpaca_account.get("portfolio_value") or alpaca_account.get("cash") or alpaca_account.get("buying_power")

        # Update if already exists, or create new
        account, created = BrokerAccount.objects.update_or_create(
            portfolio=portfolio,
            provider=provider,
            defaults={
                "credential": credential,
                "environment": env,
                "execution_mode": validated_data.get("execution_mode", "auto"),
                "display_name": validated_data["display_name"],
                "broker_label": validated_data.get("broker_label", "Alpaca"),
                "external_account_id": external_id,
                "managed_capital": portfolio_value,
                "order_policy": {
                    "cash": alpaca_account.get("cash"),
                    "buying_power": alpaca_account.get("buying_power"),
                    "currency": alpaca_account.get("currency", "USD"),
                    "status": alpaca_account.get("status"),
                },
                "trading_enabled": True,
            }
        )

        # Sync portfolio capital to match real broker equity
        if portfolio_value and float(portfolio_value) > 0:
            portfolio.paper_initial_capital = portfolio_value
            portfolio.save(update_fields=["paper_initial_capital"])

        return account
