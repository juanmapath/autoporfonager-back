from rest_framework import serializers
from apps.strategies.models import Strategy, StrategyVersion, StrategyInstrument, BacktestRun


class StrategyInstrumentSerializer(serializers.ModelSerializer):
    symbol = serializers.CharField(source="instrument.symbol", read_only=True)

    class Meta:
        model = StrategyInstrument
        fields = ["id", "symbol", "role", "weight", "params"]


class StrategyVersionSerializer(serializers.ModelSerializer):
    instruments = StrategyInstrumentSerializer(many=True, read_only=True)

    class Meta:
        model = StrategyVersion
        fields = ["id", "version", "params", "metrics", "status", "instruments"]


class StrategySerializer(serializers.ModelSerializer):
    versions = StrategyVersionSerializer(many=True, read_only=True)
    live_version = serializers.SerializerMethodField()
    current_signals = serializers.SerializerMethodField()

    class Meta:
        model = Strategy
        fields = [
            "id", "slug", "name", "family", "kind", "engine",
            "timeframe", "decision_offset_minutes",
            "is_active", "signal_only", "versions", "live_version",
            "current_signals",
        ]

    def get_live_version(self, obj):
        v = obj.versions.filter(status="live").first()
        return StrategyVersionSerializer(v).data if v else None

    def get_current_signals(self, obj):
        from apps.strategies.models import Signal
        v = obj.versions.filter(status="live").first()
        if not v:
            return []
        signals = []
        for inst in v.instruments.all().select_related("instrument"):
            sig = Signal.objects.filter(version=v, instrument=inst.instrument).order_by("-as_of").first()
            if sig:
                exp = float(sig.target_exposure)
                direction = "LONG" if exp > 0 else ("SHORT" if exp < 0 else "FLAT")
                signals.append({
                    "symbol": inst.instrument.symbol,
                    "target_exposure": exp,
                    "direction": direction,
                    "as_of": sig.as_of,
                    "changed": sig.changed,
                    "diagnostics": sig.diagnostics,
                })
        return signals


class StrategyConfigItemSerializer(serializers.Serializer):
    strategy_name = serializers.CharField(max_length=64)
    params = serializers.ListField(child=serializers.FloatField(), required=False, default=list)


class BotCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=128)
    slug = serializers.SlugField(max_length=64, required=False, allow_blank=True)
    family = serializers.CharField(max_length=64, default="General")
    bot_type = serializers.ChoiceField(choices=Strategy.KIND_CHOICES)
    traded_symbol = serializers.CharField(max_length=32, required=False, allow_blank=True)
    signal_symbol = serializers.CharField(max_length=32, required=False, allow_blank=True)
    traded_symbols = serializers.ListField(child=serializers.CharField(max_length=32), required=False, default=list)
    strategies = StrategyConfigItemSerializer(many=True, required=False, default=list)
    metrics = serializers.ListField(child=serializers.DictField(), required=False, default=list)
    rank_weights = serializers.ListField(child=serializers.FloatField(), required=False, default=list)
    rebalance_frequency = serializers.CharField(max_length=32, required=False, default="monthly")
    execution = serializers.DictField(required=False, default=dict)
    leverage = serializers.FloatField(default=1.0)
    max_leverage = serializers.FloatField(default=1.0)
    use_regimes = serializers.BooleanField(default=False)
    decision_offset_minutes = serializers.IntegerField(default=25)

    def validate(self, data):
        bot_type = data.get("bot_type")
        strategies = data.get("strategies", [])
        traded_symbol = data.get("traded_symbol")
        signal_symbol = data.get("signal_symbol")
        traded_symbols = data.get("traded_symbols", [])

        if bot_type in ("one_strategy", "multi_strategy", "cross_asset", "smart_accumulator"):
            if not traded_symbol:
                raise serializers.ValidationError({"traded_symbol": "El símbolo a operar es obligatorio para este tipo de bot."})

        if bot_type == "one_strategy" and len(strategies) != 1:
            raise serializers.ValidationError({"strategies": "Un 'One Strategy Bot' debe tener exactamente 1 estrategia seleccionada."})

        if bot_type == "multi_strategy" and len(strategies) < 2:
            raise serializers.ValidationError({"strategies": "Un 'Multi Strategy Bot' debe tener al menos 2 estrategias seleccionadas."})

        if bot_type == "cross_asset":
            if not signal_symbol:
                raise serializers.ValidationError({"signal_symbol": "Para bots Cross Asset, debes especificar el símbolo de referencia/señales (ej. TLT, QQQ, ^VIX)."})
            if len(strategies) != 1:
                raise serializers.ValidationError({"strategies": "Un 'Cross Asset Bot' debe tener 1 estrategia seleccionada para evaluar sobre el activo de señales."})

        if bot_type == "smart_accumulator" and len(strategies) != 1:
            raise serializers.ValidationError({"strategies": "Un 'Smart Accumulator' requiere 1 estrategia seleccionada para disparar las compras."})

        if bot_type == "hold":
            is_cash = data.get("is_cash", False) or (traded_symbol or "").upper() in ("CASH", "USD")
            if not is_cash and not traded_symbol:
                raise serializers.ValidationError({"traded_symbol": "Debes especificar el activo a mantener o marcar 'Reserva de Cash'."})

        if bot_type == "ranked_allocation":
            # Multi-asset validation
            symbols = traded_symbols or ([traded_symbol] if traded_symbol else [])
            if len(symbols) < 2:
                raise serializers.ValidationError({"traded_symbols": "Un 'Ranked Allocation Bot' debe contener al menos 2 activos."})

            metrics = data.get("metrics", [])
            if not metrics:
                raise serializers.ValidationError({"metrics": "Debes especificar al menos 1 métrica para el ranking."})

            rank_weights = data.get("rank_weights", [])
            if rank_weights:
                if len(rank_weights) != len(symbols):
                    raise serializers.ValidationError({"rank_weights": f"Debes asignar exactamente {len(symbols)} ponderaciones (una por posición del ranking)."})
                total_w = sum(rank_weights)
                if abs(total_w - 1.0) > 0.001:
                    raise serializers.ValidationError({"rank_weights": f"La suma de las ponderaciones de ranking debe ser 100% (suma actual: {round(total_w*100, 2)}%)."})

            exec_cfg = data.get("execution", {})
            if exec_cfg.get("mode") == "strategy":
                strat_name = exec_cfg.get("strategy_name")
                if not strat_name:
                    raise serializers.ValidationError({"execution": "En modo 'strategy', debes especificar 'strategy_name'."})

        return data

    def create(self, validated_data):
        from django.utils.text import slugify
        from apps.instruments.models import Instrument

        name = validated_data["name"]
        slug = validated_data.get("slug") or slugify(name)
        base_slug = slug
        counter = 1
        while Strategy.objects.filter(slug=slug).exists():
            slug = f"{base_slug}-{counter}"
            counter += 1

        bot_type = validated_data["bot_type"]
        engine_map = {
            "one_strategy": "single_signal",
            "multi_strategy": "multi_signal",
            "cross_asset": "cross_asset",
            "ranked_allocation": "ranked_allocation",
            "hold": "hold",
            "smart_accumulator": "smart_accumulator",
        }
        engine = engine_map.get(bot_type, "single_signal")

        strategy = Strategy.objects.create(
            name=name,
            slug=slug,
            family=validated_data.get("family", "General"),
            kind=bot_type,
            engine=engine,
            decision_offset_minutes=validated_data.get("decision_offset_minutes", 25),
            signal_only=False,
            is_active=True,
        )

        version_params = {
            "bot_type": bot_type,
            "strategies": validated_data.get("strategies", []),
            "metrics": validated_data.get("metrics", []),
            "rank_weights": validated_data.get("rank_weights", []),
            "rebalance_frequency": validated_data.get("rebalance_frequency", "monthly"),
            "execution": validated_data.get("execution", {"mode": "hold"}),
            "participates_in_rebalance": validated_data.get("participates_in_rebalance", True),
            "is_cash": validated_data.get("is_cash", False) or (validated_data.get("traded_symbol") or "").upper() in ("CASH", "USD"),
            "leverage": validated_data.get("leverage", 1.0),
            "max_leverage": validated_data.get("max_leverage", 1.0),
            "use_regimes": validated_data.get("use_regimes", False),
        }

        version = StrategyVersion.objects.create(
            strategy=strategy,
            version=1,
            params=version_params,
            status="live",
        )

        # Attach instruments
        if bot_type == "ranked_allocation":
            symbols = validated_data.get("traded_symbols") or [validated_data.get("traded_symbol")]
            for s in symbols:
                if not s:
                    continue
                inst, _ = Instrument.objects.get_or_create(
                    symbol=s.upper(),
                    defaults={"name": s.upper(), "asset_class": "equity"}
                )
                StrategyInstrument.objects.create(
                    version=version,
                    instrument=inst,
                    role="traded",
                    weight=1.0,
                )
        else:
            traded_symbol = validated_data.get("traded_symbol")
            if traded_symbol:
                traded_inst, _ = Instrument.objects.get_or_create(
                    symbol=traded_symbol.upper(),
                    defaults={"name": traded_symbol.upper(), "asset_class": "etf"}
                )
                StrategyInstrument.objects.create(
                    version=version,
                    instrument=traded_inst,
                    role="traded",
                    weight=1.0,
                )

        signal_symbol = validated_data.get("signal_symbol")
        if signal_symbol and bot_type == "cross_asset":
            sig_inst, _ = Instrument.objects.get_or_create(
                symbol=signal_symbol.upper(),
                defaults={"name": signal_symbol.upper(), "asset_class": "etf"}
            )
            StrategyInstrument.objects.create(
                version=version,
                instrument=sig_inst,
                role="signal_source",
                weight=1.0,
            )

        return strategy

    def reconfigure(self, strategy, validated_data):
        from apps.instruments.models import Instrument

        if "name" in validated_data and validated_data["name"]:
            strategy.name = validated_data["name"]
        if "family" in validated_data:
            strategy.family = validated_data["family"]
        if "decision_offset_minutes" in validated_data:
            strategy.decision_offset_minutes = validated_data["decision_offset_minutes"]
        strategy.save()

        # Increment version
        last_ver = strategy.versions.order_by("-version").first()
        next_version_num = (last_ver.version + 1) if last_ver else 1

        # Retire older live versions
        strategy.versions.filter(status="live").update(status="retired")

        bot_type = validated_data.get("bot_type", strategy.kind)
        version_params = {
            "bot_type": bot_type,
            "strategies": validated_data.get("strategies", []),
            "metrics": validated_data.get("metrics", []),
            "rank_weights": validated_data.get("rank_weights", []),
            "rebalance_frequency": validated_data.get("rebalance_frequency", "monthly"),
            "execution": validated_data.get("execution", {"mode": "hold"}),
            "participates_in_rebalance": validated_data.get("participates_in_rebalance", True),
            "is_cash": validated_data.get("is_cash", False) or (validated_data.get("traded_symbol") or "").upper() in ("CASH", "USD"),
            "leverage": validated_data.get("leverage", 1.0),
            "max_leverage": validated_data.get("max_leverage", 1.0),
            "use_regimes": validated_data.get("use_regimes", False),
        }

        new_version = StrategyVersion.objects.create(
            strategy=strategy,
            version=next_version_num,
            params=version_params,
            status="live",
        )

        if bot_type == "ranked_allocation":
            symbols = validated_data.get("traded_symbols") or [validated_data.get("traded_symbol")]
            for s in symbols:
                if not s:
                    continue
                inst, _ = Instrument.objects.get_or_create(
                    symbol=s.upper(),
                    defaults={"name": s.upper(), "asset_class": "equity"}
                )
                StrategyInstrument.objects.create(
                    version=new_version,
                    instrument=inst,
                    role="traded",
                    weight=1.0,
                )
        else:
            traded_symbol = validated_data.get("traded_symbol")
            if traded_symbol:
                traded_inst, _ = Instrument.objects.get_or_create(
                    symbol=traded_symbol.upper(),
                    defaults={"name": traded_symbol.upper(), "asset_class": "etf"}
                )
                StrategyInstrument.objects.create(
                    version=new_version,
                    instrument=traded_inst,
                    role="traded",
                    weight=1.0,
                )

        signal_symbol = validated_data.get("signal_symbol")
        if signal_symbol and bot_type == "cross_asset":
            sig_inst, _ = Instrument.objects.get_or_create(
                symbol=signal_symbol.upper(),
                defaults={"name": signal_symbol.upper(), "asset_class": "etf"}
            )
            StrategyInstrument.objects.create(
                version=new_version,
                instrument=sig_inst,
                role="signal_source",
                weight=1.0,
            )

        return strategy


class BacktestRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = BacktestRun
        fields = "__all__"
