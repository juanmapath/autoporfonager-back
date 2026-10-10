from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
import numpy as np
import pandas as pd
from django.utils import timezone

from apps.strategies.signals.catalog import STRATEGY_MAP


def fetch_historical_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Descarga barras históricas de Yahoo Finance con periodo de calentamiento previo
    para que medias móviles largas y Z-Scores tengan datos válidos desde el día 1.
    """
    import yfinance as yf

    sym = symbol.strip().upper()
    dt_start = pd.to_datetime(start_date)
    dt_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    pad_start = (dt_start - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    end_str = dt_end.strftime("%Y-%m-%d")

    try:
        df = yf.download(sym, start=pad_start, end=end_str, progress=False, auto_adjust=True)
    except Exception as e:
        raise ValueError(f"Error al descargar datos de Yahoo Finance para '{sym}': {str(e)}")

    if df is None or df.empty:
        raise ValueError(f"No se encontraron datos históricos para el ticker '{sym}'. Verifica que el símbolo exista en Yahoo Finance (ej. QQQ, SPY, NVDA, AAPL, BTC-USD).")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols_needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Columna faltante '{c}' en datos de {sym}")

    df = df[cols_needed].copy().ffill().bfill()
    df.index = pd.to_datetime(df.index)
    return df


class BacktestEngine:
    """
    Motor cuantitativo de backtesting profesional para bots de trading algorítmico.
    Garantiza 100% de paridad con las señales de producción y calcula métricas
    financieras completas (ROI, CAGR, Max DD, Sharpe, Sortino, Win Rate, Profit Factor,
    Apalancamiento Dinámico Walk-Forward y atribución granular por sub-estrategia).
    """

    def __init__(self, initial_cash: float = 100000.0, commission_per_trade: float = 0.0, slippage_pct: float = 0.0005):
        self.initial_cash = float(initial_cash)
        self.commission = float(commission_per_trade)
        self.slippage = float(slippage_pct)

    def run(self, df: pd.DataFrame, strategy_name: str, params: Any = None) -> Dict[str, Any]:
        """Convenience method to run a simulation directly on an existing DataFrame."""
        strat_func = STRATEGY_MAP.get(strategy_name)
        if not strat_func:
            raise ValueError(f"Estrategia '{strategy_name}' no encontrada en el catálogo.")
        sig, _ = strat_func(df, params if params is not None else [])
        state = pd.Series(0, index=sig.index)
        curr = 0
        for i in range(len(sig)):
            v = sig.iloc[i]
            if v == 1:
                curr = 1
            elif v == -1:
                curr = 0
            state.iloc[i] = curr
        return self._simulate(df=df, signals=state, symbol="ASSET")

    def run_for_strategy(
        self,
        strategy,
        start_date: str,
        end_date: str,
        leverage: Optional[float] = None,
        max_leverage: Optional[float] = None,
        use_dynamic_leverage: Optional[bool] = None,
        periodic_deposit: Optional[float] = None,
        deposit_frequency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el backtest histórico completo de un Bot (Strategy) de la base de datos.
        """
        version = strategy.versions.filter(status="live").first() or strategy.versions.order_by("-version").first()
        if not version:
            raise ValueError(f"El bot '{strategy.name}' no tiene versiones de estrategia registradas.")

        params = version.params or {}
        instruments = list(version.instruments.all().select_related("instrument"))
        traded_inst = next((i for i in instruments if i.role == "traded"), None)
        signal_inst = next((i for i in instruments if i.role == "signal_source"), None)

        if not traded_inst:
            if instruments:
                traded_inst = instruments[0]
            else:
                raise ValueError(f"El bot '{strategy.name}' no tiene instrumentos configurados.")

        traded_sym = traded_inst.instrument.symbol
        signal_sym = signal_inst.instrument.symbol if signal_inst else traded_sym
        raw_strats = params.get("strategies", [])
        bot_type = params.get("bot_type") or strategy.kind or "one_strategy"

        strat_base_lev = float(leverage if leverage is not None else params.get("leverage", 1.0))
        strat_max_lev = float(max_leverage if max_leverage is not None else params.get("max_leverage", strat_base_lev))
        strat_dyn = use_dynamic_leverage if use_dynamic_leverage is not None else params.get("use_regimes", False)
        strat_deposit = float(periodic_deposit if periodic_deposit is not None else params.get("periodic_deposit", 0.0))
        strat_freq = str(deposit_frequency or params.get("deposit_frequency", "monthly"))

        return self.run_for_config(
            bot_type=bot_type,
            traded_symbol=traded_sym,
            signal_symbol=signal_sym,
            strategies=raw_strats,
            start_date=start_date,
            end_date=end_date,
            leverage=strat_base_lev,
            max_leverage=strat_max_lev,
            use_dynamic_leverage=strat_dyn,
            strategy_params=params,
            periodic_deposit=strat_deposit,
            deposit_frequency=strat_freq,
        )

    def run_for_config(
        self,
        bot_type: str,
        traded_symbol: str,
        signal_symbol: Optional[str] = None,
        strategies: Optional[List[Dict[str, Any]]] = None,
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        leverage: float = 1.0,
        max_leverage: float = 1.0,
        use_dynamic_leverage: bool = False,
        strategy_params: Optional[Dict[str, Any]] = None,
        periodic_deposit: float = 0.0,
        deposit_frequency: str = "monthly",
    ) -> Dict[str, Any]:
        """
        Ejecuta el backtest a partir de una configuración ad-hoc o personalizada.
        Permite probar cualquier ticker, tipo de bot, combinación de estrategias,
        acumulación inteligente DCA con aportes periódicos y apalancamiento dinámico.
        """
        traded_sym = str(traded_symbol).strip().upper()
        if not traded_sym:
            raise ValueError("Debes especificar un ticker para el activo negociado.")

        signal_sym = str(signal_symbol).strip().upper() if signal_symbol else traded_sym
        if not signal_sym:
            signal_sym = traded_sym

        # 1. Ingesta de datos
        df_traded = fetch_historical_ohlcv(traded_sym, start_date, end_date)
        df_signal = fetch_historical_ohlcv(signal_sym, start_date, end_date) if signal_sym != traded_sym else df_traded

        # 2. Generación de Señales y Atribución por Sub-Estrategia
        signals_series = pd.Series(0, index=df_traded.index)
        strategies = strategies or []
        strategy_params = strategy_params or {}

        entry_meta_by_date: Dict[str, List[str]] = {}
        exit_meta_by_date: Dict[str, List[str]] = {}

        if bot_type == "ranked_allocation":
            raise ValueError("El backtest no está disponible para bots Ranked Allocation (datos fundamentales históricos insuficientes).")
        if bot_type == "hold":
            raise ValueError("El backtest no está disponible para bots Hold & Reserve (estrategia de custodia pasiva).")

        if bot_type == "smart_accumulator":
            all_signals = []
            if strategies:
                for item in strategies:
                    if isinstance(item, dict):
                        s_name = item.get("strategy_name")
                        s_param = item.get("params", [])
                    elif isinstance(item, str):
                        s_name = item
                        s_param = []
                    else:
                        continue

                    strat_func = STRATEGY_MAP.get(s_name)
                    if strat_func:
                        sig, _ = strat_func(df_signal, s_param)
                        all_signals.append((sig, s_name))

            if not all_signals and strategy_params:
                s_name = strategy_params.get("strategy_name") or strategy_params.get("strategy_function")
                s_param = strategy_params.get("strategy_params", strategy_params.get("params", {}))
                strat_func = STRATEGY_MAP.get(s_name)
                if strat_func:
                    sig, _ = strat_func(df_signal, s_param)
                    all_signals.append((sig, s_name))

            if not all_signals:
                raise ValueError("Un Smart Accumulator requiere al menos una estrategia cuantitativa para disparar las compras.")

            signals_series = pd.Series(0, index=df_signal.index)
            for idx_i, dt in enumerate(df_traded.index):
                d_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
                active_entries = []
                for sig, s_name in all_signals:
                    if sig.iloc[idx_i] == 1:
                        active_entries.append(s_name)
                if active_entries:
                    signals_series.iloc[idx_i] = 1
                    entry_meta_by_date[d_str] = [f"Acumular ({', '.join(active_entries)})"]

        elif bot_type == "multi_strategy" or len(strategies) > 1:
            all_signals = []
            all_names = []
            for item in strategies:
                if isinstance(item, dict):
                    s_name = item.get("strategy_name")
                    s_param = item.get("params", [])
                elif isinstance(item, str):
                    s_name = item
                    s_param = []
                else:
                    continue

                strat_func = STRATEGY_MAP.get(s_name)
                if strat_func:
                    sig, _ = strat_func(df_signal, s_param)
                    all_signals.append(sig)
                    all_names.append(s_name)

            if all_signals:
                active_states = []
                for sig in all_signals:
                    state = pd.Series(False, index=sig.index)
                    curr = False
                    for i in range(len(sig)):
                        v = sig.iloc[i]
                        if v == 1:
                            curr = True
                        elif v == -1:
                            curr = False
                        state.iloc[i] = curr
                    active_states.append(state)

                combined_long = pd.Series(False, index=df_signal.index)
                for state in active_states:
                    combined_long = combined_long | state

                signals_series = np.where(combined_long, 1, 0)
                signals_series = pd.Series(signals_series, index=df_signal.index)

                # Atribución precisa: ¿qué estrategia disparó la entrada o salida en cada vela?
                n_bars = len(df_signal)
                for i in range(n_bars):
                    dt_val = df_signal.index[i]
                    d_str = str(dt_val.date()) if hasattr(dt_val, "date") else str(dt_val)

                    entry_strats = []
                    exit_strats = []
                    for idx_s, sig in enumerate(all_signals):
                        # Señal de compra directa o cambio de estado
                        if sig.iloc[i] == 1:
                            entry_strats.append(all_names[idx_s])
                        elif sig.iloc[i] == -1:
                            exit_strats.append(all_names[idx_s])

                    if entry_strats:
                        entry_meta_by_date[d_str] = entry_strats
                    if exit_strats:
                        exit_meta_by_date[d_str] = exit_strats

        elif bot_type == "cross_asset":
            s_name = None
            s_param = {}
            if strategies:
                s_name = strategies[0].get("strategy_name")
                s_param = strategies[0].get("params", {})
            if not s_name:
                s_name = strategy_params.get("strategy_name")
                s_param = strategy_params.get("params", {})

            strat_func = STRATEGY_MAP.get(s_name)
            if not strat_func:
                raise ValueError(f"Estrategia '{s_name}' no encontrada en el catálogo.")

            sig, _ = strat_func(df_signal, s_param)
            # Convertir pulsos discretos (1, -1, 0) a estado continuo de posición (1: dentro, 0: fuera)
            state = pd.Series(0, index=sig.index)
            curr = 0
            for i in range(len(sig)):
                v = sig.iloc[i]
                if v == 1:
                    curr = 1
                elif v == -1:
                    curr = 0
                state.iloc[i] = curr
            signals_series = state

            label = f"{s_name} ({signal_sym})"
            for idx_i, dt in enumerate(df_traded.index):
                d_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
                if sig.iloc[idx_i] == 1:
                    entry_meta_by_date[d_str] = [label]
                elif sig.iloc[idx_i] == -1:
                    exit_meta_by_date[d_str] = [label]

        else:
            # Single Strategy (one_strategy)
            s_name = None
            s_param = {}
            if strategies:
                first = strategies[0]
                if isinstance(first, dict):
                    s_name = first.get("strategy_name")
                    s_param = first.get("params", {})
                elif isinstance(first, str):
                    s_name = first
            if not s_name:
                s_name = strategy_params.get("strategy_name") or strategy_params.get("strategy_function")
                s_param = strategy_params.get("strategy_params", strategy_params.get("params", {}))

            if not s_name:
                raise ValueError("No se especificó ninguna función de estrategia para evaluar.")

            strat_func = STRATEGY_MAP.get(s_name)
            if not strat_func:
                raise ValueError(f"Estrategia '{s_name}' no encontrada en el catálogo.")

            sig, _ = strat_func(df_signal, s_param)
            # Convertir pulsos discretos (1, -1, 0) a estado continuo de posición (1: dentro, 0: fuera)
            state = pd.Series(0, index=sig.index)
            curr = 0
            for i in range(len(sig)):
                v = sig.iloc[i]
                if v == 1:
                    curr = 1
                elif v == -1:
                    curr = 0
                state.iloc[i] = curr
            signals_series = state

            for idx_i, dt in enumerate(df_traded.index):
                d_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
                if sig.iloc[idx_i] == 1:
                    entry_meta_by_date[d_str] = [s_name]
                elif sig.iloc[idx_i] == -1:
                    exit_meta_by_date[d_str] = [s_name]

        # Recortar al rango solicitado (eliminando el padding previo de calentamiento)
        dt_start_pd = pd.to_datetime(start_date)
        mask_range = df_traded.index >= dt_start_pd
        df_traded_eval = df_traded[mask_range].copy()
        signals_eval = signals_series.reindex(df_traded_eval.index).fillna(0)

        if bot_type == "smart_accumulator":
            return self._simulate_accumulator(
                df=df_traded_eval,
                signals=signals_eval,
                symbol=traded_sym,
                periodic_deposit=float(periodic_deposit),
                deposit_frequency=str(deposit_frequency),
                entry_meta_by_date=entry_meta_by_date,
            )

        # 3. Simulación de Ejecución con Apalancamiento Dinámico y Atribución
        return self._simulate(
            df=df_traded_eval,
            signals=signals_eval,
            symbol=traded_sym,
            base_leverage=float(leverage),
            max_leverage=float(max_leverage if max_leverage >= leverage else leverage),
            use_dynamic_leverage=bool(use_dynamic_leverage),
            entry_meta_by_date=entry_meta_by_date,
            exit_meta_by_date=exit_meta_by_date,
        )

    def _simulate_accumulator(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str,
        periodic_deposit: float = 0.0,
        deposit_frequency: str = "monthly",
        entry_meta_by_date: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Simulación cuantitativa de Dollar-Cost-Averaging (DCA) Inteligente.
        - Inyecta aportes periódicos de capital (semanal o mensual).
        - Despliega capital en compras SOLO cuando la estrategia genera señal de compra (1).
        - No vende en señales neutras o de salida (mantiene acciones acumuladas).
        - Evalúa retornos sobre el capital total aportado.
        - Compara contra un DCA Benchmark pasivo (inversión automática sistemática en cada fecha de aporte).
        """
        closes = df["Close"].values
        dates = df.index
        n = len(closes)

        if n < 2:
            raise ValueError("Datos insuficientes en el rango para ejecutar la simulación de acumulación.")

        # --- Benchmark: DCA Pasivo Sistemático ---
        # El benchmark compra con el capital inicial el día 1, y compra con cada aporte en la fecha correspondiente
        bh_first_price = float(closes[0]) * (1.0 + self.slippage)
        bh_shares = (self.initial_cash - self.commission) / bh_first_price if bh_first_price > 0 and self.initial_cash > 0 else 0.0
        bh_portfolio_values = np.zeros(n)

        # --- Smart Accumulator Strategy ---
        cash = self.initial_cash
        total_invested = self.initial_cash
        shares = 0.0
        deposits_count = 0
        trades = []
        trade_markers = []
        portfolio_values = np.zeros(n)
        invested_curve = np.zeros(n)

        for i in range(n):
            price = float(closes[i])
            curr_dt = dates[i]
            d_str = str(curr_dt.date()) if hasattr(curr_dt, "date") else str(curr_dt)

            # Inyección periódica de capital (a partir de la barra 1)
            if i > 0 and periodic_deposit > 0:
                prev_dt = dates[i - 1]
                is_deposit = False
                if deposit_frequency == "weekly":
                    is_deposit = curr_dt.isocalendar()[1] != prev_dt.isocalendar()[1]
                else:  # monthly default
                    is_deposit = curr_dt.month != prev_dt.month or curr_dt.year != prev_dt.year

                if is_deposit:
                    cash += periodic_deposit
                    total_invested += periodic_deposit
                    deposits_count += 1

                    # Benchmark DCA pasivo compra en cada inyección
                    bench_adj_p = price * (1.0 + self.slippage)
                    if bench_adj_p > 0:
                        bh_shares += max(0.0, (periodic_deposit - self.commission) / bench_adj_p)

            # Evaluación de Señal de Compra / Acumulación
            sig = int(signals.iloc[i]) if pd.notnull(signals.iloc[i]) else 0
            if sig == 1 and cash >= 10.0:
                adj_price = price * (1.0 + self.slippage)
                investable = cash - self.commission
                if investable > 0 and adj_price > 0:
                    new_shares = investable / adj_price
                    cash -= (new_shares * adj_price) + self.commission
                    shares += new_shares

                    matched = entry_meta_by_date.get(d_str) if entry_meta_by_date else None
                    strat_name = " + ".join(matched) if matched else "Señal de Acumulación"

                    trade_item = {
                        "entry_date": d_str,
                        "exit_date": str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1]),
                        "entry_price": round(adj_price, 2),
                        "exit_price": round(float(closes[-1]), 2),
                        "shares": round(new_shares, 4),
                        "invested_amount": round(investable, 2),
                        "total_shares": round(shares, 4),
                        "cash_remaining": round(cash, 2),
                        "entry_strategy": strat_name,
                        "exit_strategy": "Acumulado (Hold)",
                        "pnl": 0.0,
                        "pnl_pct": 0.0,
                        "leverage": 1.0,
                        "open": True,
                    }
                    trades.append(trade_item)

                    trade_markers.append({
                        "date": d_str,
                        "type": "BUY",
                        "price": round(adj_price, 2),
                        "strategy": strat_name,
                        "leverage": 1.0,
                    })

            portfolio_values[i] = cash + (shares * price)
            bh_portfolio_values[i] = bh_shares * price
            invested_curve[i] = total_invested

        # Cálculo de PnL final de cada lote acumulado
        last_price = float(closes[-1])
        for t in trades:
            unrealized = (t["shares"] * last_price) - t["invested_amount"]
            pct = ((last_price - t["entry_price"]) / t["entry_price"]) * 100 if t["entry_price"] > 0 else 0.0
            t["pnl"] = round(unrealized, 2)
            t["pnl_pct"] = round(pct, 2)

        pv_series = pd.Series(portfolio_values, index=dates)
        bh_series = pd.Series(bh_portfolio_values, index=dates)

        final_val = float(pv_series.iloc[-1])
        bh_final_val = float(bh_series.iloc[-1])

        total_roi = ((final_val - total_invested) / total_invested) * 100 if total_invested > 0 else 0.0
        bh_roi = ((bh_final_val - total_invested) / total_invested) * 100 if total_invested > 0 else 0.0

        # Drawdowns
        cum_max = pv_series.cummax()
        drawdown = (pv_series - cum_max) / cum_max
        max_dd = float(drawdown.min() * 100) if not drawdown.empty else 0.0

        bh_cum_max = bh_series.cummax()
        bh_drawdown = (bh_series - bh_cum_max) / bh_cum_max
        bh_max_dd = float(bh_drawdown.min() * 100) if not bh_drawdown.empty else 0.0

        # CAGR
        days = (dates[-1] - dates[0]).days if hasattr(dates[0], "day") else len(dates)
        years = max(days / 365.25, 0.1)
        cagr = (((final_val / total_invested) ** (1 / years)) - 1) * 100 if final_val > 0 and total_invested > 0 else 0.0
        bh_cagr = (((bh_final_val / total_invested) ** (1 / years)) - 1) * 100 if bh_final_val > 0 and total_invested > 0 else 0.0

        # Sharpe & Sortino
        daily_ret = pv_series.pct_change().dropna()
        std_ret = daily_ret.std()
        sharpe = float((daily_ret.mean() / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0

        downside = daily_ret[daily_ret < 0]
        downside_std = np.sqrt(np.mean(downside**2)) if not downside.empty else 0.0
        sortino = float((daily_ret.mean() / downside_std) * np.sqrt(252)) if downside_std > 0 else 0.0

        # Trade Stats (Lotes en ganancia)
        n_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0.0
        total_gain = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = round(total_gain / total_loss, 2) if total_loss > 0 else (99.0 if total_gain > 0 else 0.0)

        # Equity Curve Samples
        step = max(1, len(pv_series) // 160)
        curve = []
        for i in range(0, len(pv_series), step):
            dt_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])
            curve.append({
                "date": dt_str,
                "strategy_value": round(float(pv_series.iloc[i]), 2),
                "benchmark_value": round(float(bh_series.iloc[i]), 2),
                "invested_capital": round(float(invested_curve[i]), 2),
            })

        last_dt_str = str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1])
        if not curve or curve[-1]["date"] != last_dt_str:
            curve.append({
                "date": last_dt_str,
                "strategy_value": round(final_val, 2),
                "benchmark_value": round(bh_final_val, 2),
                "invested_capital": round(float(total_invested), 2),
            })

        return {
            "symbol": symbol,
            "bot_type": "smart_accumulator",
            "initial_cash": self.initial_cash,
            "total_invested": round(total_invested, 2),
            "periodic_deposit": round(float(periodic_deposit), 2),
            "deposit_frequency": deposit_frequency,
            "deposits_count": deposits_count,
            "accumulated_shares": round(shares, 4),
            "cash_remaining": round(cash, 2),
            "final_value": round(final_val, 2),
            "benchmark_final_value": round(bh_final_val, 2),
            "roi_pct": round(total_roi, 2),
            "benchmark_roi_pct": round(bh_roi, 2),
            "cagr_pct": round(cagr, 2),
            "benchmark_cagr_pct": round(bh_cagr, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "benchmark_max_drawdown_pct": round(bh_max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": profit_factor,
            "alpha_vs_benchmark": round(total_roi - bh_roi, 2),
            "equity_curve": curve,
            "trades": trades,
            "trade_markers": trade_markers,
            "applied_leverage": {
                "base_leverage": 1.0,
                "max_leverage": 1.0,
                "use_dynamic_leverage": False,
            },
        }

    def _simulate(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str,
        base_leverage: float = 1.0,
        max_leverage: float = 1.0,
        use_dynamic_leverage: bool = False,
        entry_meta_by_date: Optional[Dict[str, List[str]]] = None,
        exit_meta_by_date: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        closes = df["Close"].values
        dates = df.index
        n = len(closes)

        if n < 2:
            raise ValueError("Datos insuficientes en el rango para ejecutar la simulación.")

        # --- Benchmark: Buy & Hold ---
        bh_first_price = float(closes[0])
        bh_shares = self.initial_cash / bh_first_price if bh_first_price > 0 else 0.0
        bh_portfolio_values = bh_shares * closes

        # --- Trading Strategy ---
        cash = self.initial_cash
        shares = 0.0
        portfolio_values = np.zeros(n)
        trades = []
        entry_price = 0.0
        entry_date = None
        entry_strat_name = "Estrategia"
        trade_leverage = base_leverage
        in_position = False

        for i in range(n):
            price = float(closes[i])
            sig = int(signals.iloc[i]) if pd.notnull(signals.iloc[i]) else 0
            d_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])

            # Buy Signal (1)
            if sig == 1 and not in_position:
                # 1. Determinar Apalancamiento Dinámico según Profit Factor Walk-Forward
                if use_dynamic_leverage and len(trades) >= 2:
                    gains = [t["pnl"] for t in trades if t["pnl"] > 0]
                    losses = [abs(t["pnl"]) for t in trades if t["pnl"] <= 0]
                    sum_gains = sum(gains)
                    sum_losses = sum(losses)
                    running_pf = sum_gains / sum_losses if sum_losses > 0 else (99.0 if sum_gains > 0 else 1.0)

                    # Escalamiento adaptativo
                    if running_pf >= 2.5:
                        active_leverage = max_leverage
                    elif running_pf >= 1.5:
                        active_leverage = base_leverage + 0.5 * (max_leverage - base_leverage)
                    else:
                        active_leverage = base_leverage
                else:
                    active_leverage = base_leverage

                active_leverage = max(1.0, float(active_leverage))
                adj_price = price * (1.0 + self.slippage)
                total_equity = cash
                investable = (total_equity * active_leverage) - self.commission

                if investable > 0 and adj_price > 0:
                    shares = investable / adj_price
                    cash -= (shares * adj_price) + self.commission
                    entry_price = adj_price
                    entry_date = d_str
                    trade_leverage = active_leverage

                    # Atribución de inicio
                    matched_entry = entry_meta_by_date.get(d_str) if entry_meta_by_date else None
                    entry_strat_name = " + ".join(matched_entry) if matched_entry else "Estrategia"
                    in_position = True

            # Sell / Flat Signal (-1 or 0 when in_position)
            elif (sig == -1 or sig == 0) and in_position:
                adj_price = price * (1.0 - self.slippage)
                proceeds = (shares * adj_price) - self.commission
                cash += proceeds
                pnl = proceeds - (shares * entry_price)
                pnl_pct = ((adj_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
                exit_date = d_str

                # Atribución de salida
                matched_exit = exit_meta_by_date.get(d_str) if exit_meta_by_date else None
                exit_strat_name = " + ".join(matched_exit) if matched_exit else "Regla de Salida"

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(adj_price, 2),
                    "shares": round(shares, 4),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "leverage": round(trade_leverage, 2),
                    "entry_strategy": entry_strat_name,
                    "exit_strategy": exit_strat_name,
                })

                shares = 0.0
                in_position = False

            curr_val = cash + (shares * price)
            portfolio_values[i] = curr_val

        pv_series = pd.Series(portfolio_values, index=dates)
        bh_series = pd.Series(bh_portfolio_values, index=dates)

        # Si terminó en posición abierta, registrar trade no realizado
        if in_position:
            last_price = float(closes[-1])
            unrealized_pnl = (shares * last_price) - (shares * entry_price)
            unrealized_pct = ((last_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
            last_date_str = str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1])
            trades.append({
                "entry_date": entry_date,
                "exit_date": last_date_str,
                "entry_price": round(entry_price, 2),
                "exit_price": round(last_price, 2),
                "shares": round(shares, 4),
                "pnl": round(unrealized_pnl, 2),
                "pnl_pct": round(unrealized_pct, 2),
                "leverage": round(trade_leverage, 2),
                "entry_strategy": entry_strat_name,
                "exit_strategy": "En Posición (Abierto)",
                "open": True,
            })

        # --- Métricas Cuantitativas ---
        final_val = float(pv_series.iloc[-1])
        bh_final_val = float(bh_series.iloc[-1])

        total_roi = ((final_val - self.initial_cash) / self.initial_cash) * 100
        bh_roi = ((bh_final_val - self.initial_cash) / self.initial_cash) * 100

        # Max Drawdown
        cum_max = pv_series.cummax()
        drawdown = (pv_series - cum_max) / cum_max
        max_dd = float(drawdown.min() * 100) if not drawdown.empty else 0.0

        bh_cum_max = bh_series.cummax()
        bh_drawdown = (bh_series - bh_cum_max) / bh_cum_max
        bh_max_dd = float(bh_drawdown.min() * 100) if not bh_drawdown.empty else 0.0

        # CAGR
        days = (dates[-1] - dates[0]).days if hasattr(dates[0], "day") else len(dates)
        years = max(days / 365.25, 0.1)
        cagr = (((final_val / self.initial_cash) ** (1 / years)) - 1) * 100 if final_val > 0 else 0.0
        bh_cagr = (((bh_final_val / self.initial_cash) ** (1 / years)) - 1) * 100 if bh_final_val > 0 else 0.0

        # Sharpe & Sortino
        daily_ret = pv_series.pct_change().dropna()
        std_ret = daily_ret.std()
        sharpe = float((daily_ret.mean() / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0

        downside = daily_ret[daily_ret < 0]
        downside_std = np.sqrt(np.mean(downside**2)) if not downside.empty else 0.0
        sortino = float((daily_ret.mean() / downside_std) * np.sqrt(252)) if downside_std > 0 else 0.0

        # Trade Stats
        n_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0.0

        total_gain = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = round(total_gain / total_loss, 2) if total_loss > 0 else (99.0 if total_gain > 0 else 0.0)

        # Equity Curve Samples (~160 puntos para graficación rápida)
        step = max(1, len(pv_series) // 160)
        curve = []
        for i in range(0, len(pv_series), step):
            dt_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])
            curve.append({
                "date": dt_str,
                "strategy_value": round(float(pv_series.iloc[i]), 2),
                "benchmark_value": round(float(bh_series.iloc[i]), 2),
            })

        # Asegurar último punto
        last_dt_str = str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1])
        if not curve or curve[-1]["date"] != last_dt_str:
            curve.append({
                "date": last_dt_str,
                "strategy_value": round(final_val, 2),
                "benchmark_value": round(bh_final_val, 2),
            })

        # Construir marcadores visuales para la gráfica
        trade_markers = []
        for t in trades:
            trade_markers.append({
                "date": t["entry_date"],
                "type": "BUY",
                "price": t["entry_price"],
                "strategy": t.get("entry_strategy", "Estrategia"),
                "leverage": t.get("leverage", 1.0),
            })
            if not t.get("open"):
                trade_markers.append({
                    "date": t["exit_date"],
                    "type": "SELL",
                    "price": t["exit_price"],
                    "strategy": t.get("exit_strategy", "Salida"),
                    "pnl": t["pnl"],
                    "pnl_pct": t["pnl_pct"],
                    "is_win": t["pnl"] > 0,
                })

        return {
            "symbol": symbol,
            "initial_cash": self.initial_cash,
            "final_value": round(final_val, 2),
            "benchmark_final_value": round(bh_final_val, 2),
            "roi_pct": round(total_roi, 2),
            "benchmark_roi_pct": round(bh_roi, 2),
            "cagr_pct": round(cagr, 2),
            "benchmark_cagr_pct": round(bh_cagr, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "benchmark_max_drawdown_pct": round(bh_max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": profit_factor,
            "alpha_vs_benchmark": round(total_roi - bh_roi, 2),
            "equity_curve": curve,
            "trades": trades,
            "trade_markers": trade_markers,
            "applied_leverage": {
                "base_leverage": base_leverage,
                "max_leverage": max_leverage,
                "use_dynamic_leverage": use_dynamic_leverage,
            },
        }
