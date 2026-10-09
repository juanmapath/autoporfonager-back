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

    dt_start = pd.to_datetime(start_date)
    dt_end = pd.to_datetime(end_date)
    pad_start = (dt_start - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    end_str = dt_end.strftime("%Y-%m-%d")

    df = yf.download(symbol, start=pad_start, end=end_str, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No se pudieron descargar datos históricos para {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols_needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Columna faltante '{c}' en datos de {symbol}")

    df = df[cols_needed].copy().ffill().bfill()
    df.index = pd.to_datetime(df.index)
    return df


class BacktestEngine:
    """
    Motor cuantitativo de backtesting profesional para bots de trading algorítmico.
    Garantiza 100% de paridad con las señales de producción y calcula métricas
    financieras completas (ROI, CAGR, Max DD, Sharpe, Sortino, Win Rate, Profit Factor).
    """

    def __init__(self, initial_cash: float = 100000.0, commission_per_trade: float = 0.0, slippage_pct: float = 0.0005):
        self.initial_cash = float(initial_cash)
        self.commission = float(commission_per_trade)
        self.slippage = float(slippage_pct)

    def run_for_strategy(
        self,
        strategy,
        start_date: str,
        end_date: str,
        leverage: float = 1.0,
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

        # 1. Ingesta de datos
        df_traded = fetch_historical_ohlcv(traded_sym, start_date, end_date)
        df_signal = fetch_historical_ohlcv(signal_sym, start_date, end_date) if signal_sym != traded_sym else df_traded

        # 2. Generación de Señales según el motor
        engine_type = strategy.engine
        signals_series = pd.Series(0, index=df_traded.index)

        if engine_type == "multi_signal" or "strategies" in params:
            # Multi-estrategia con unión OR y seguimiento continuo de estado
            raw_strategies = params.get("strategies", [])
            strategies_list = []
            strategies_params = []
            if raw_strategies and isinstance(raw_strategies[0], dict):
                for item in raw_strategies:
                    strategies_list.append(item.get("strategy_name"))
                    strategies_params.append(item.get("params", []))
            else:
                strategies_list = raw_strategies
                strategies_params = params.get("strategies_params", [])

            all_signals = []
            for s_name, s_param in zip(strategies_list, strategies_params):
                strat_func = STRATEGY_MAP.get(s_name)
                if strat_func:
                    sig, _ = strat_func(df_signal, s_param)
                    all_signals.append(sig)

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

        elif engine_type == "cross_asset":
            s_name = params.get("strategy_name")
            s_params = params.get("params", {})
            strat_func = STRATEGY_MAP.get(s_name)
            if not strat_func and "strategies" in params and params["strategies"]:
                first_s = params["strategies"][0]
                if isinstance(first_s, dict):
                    s_name = first_s.get("strategy_name")
                    s_params = first_s.get("params", s_params)
                strat_func = STRATEGY_MAP.get(s_name)

            if strat_func:
                sig, _ = strat_func(df_signal, s_params)
                signals_series = sig
            else:
                raise ValueError(f"Función de estrategia '{s_name}' no encontrada en catálogo.")

        else:
            # Single Signal Engine
            s_name = params.get("strategy_name") or params.get("strategy_function")
            s_params = params.get("strategy_params", params.get("params", {}))
            if not s_name and "strategies" in params and params["strategies"]:
                first_s = params["strategies"][0]
                if isinstance(first_s, dict):
                    s_name = first_s.get("strategy_name")
                    s_params = first_s.get("params", s_params)
                elif isinstance(first_s, str):
                    s_name = first_s

            strat_func = STRATEGY_MAP.get(s_name)
            if not strat_func:
                raise ValueError(f"Estrategia '{s_name}' no encontrada en catálogo.")
            sig, _ = strat_func(df_signal, s_params)
            signals_series = sig

        # Alinear fechas al rango solicitado (eliminando el padding inicial de calentamiento)
        dt_start_pd = pd.to_datetime(start_date)
        mask_range = df_traded.index >= dt_start_pd
        df_traded_eval = df_traded[mask_range].copy()
        signals_eval = signals_series.reindex(df_traded_eval.index).fillna(0)

        # 3. Simulación de Ejecución
        return self._simulate(df_traded_eval, signals_eval, traded_sym, leverage=leverage)

    def run(self, df: pd.DataFrame, strategy_name: str, params: Any, leverage: float = 1.0) -> Dict[str, Any]:
        """
        Ejecuta backtest directamente sobre un DataFrame para una estrategia puntual del catálogo.
        """
        strat_func = STRATEGY_MAP.get(strategy_name)
        if not strat_func:
            raise ValueError(f"Strategy '{strategy_name}' not found in STRATEGY_MAP.")

        signals, _ = strat_func(df, params)
        return self._simulate(df, signals, symbol="ASSET", leverage=leverage)

    def _simulate(self, df: pd.DataFrame, signals: pd.Series, symbol: str, leverage: float = 1.0) -> Dict[str, Any]:
        closes = df["Close"].values
        dates = df.index
        n = len(closes)

        if n < 2:
            raise ValueError("Datos insuficientes para ejecutar la simulación.")

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
        in_position = False

        for i in range(n):
            price = float(closes[i])
            sig = int(signals.iloc[i]) if pd.notnull(signals.iloc[i]) else 0

            # Buy Signal (1)
            if sig == 1 and not in_position:
                adj_price = price * (1.0 + self.slippage)
                total_equity = cash
                investable = (total_equity * leverage) - self.commission
                if investable > 0 and adj_price > 0:
                    shares = investable / adj_price
                    cash -= (shares * adj_price) + self.commission
                    entry_price = adj_price
                    entry_date = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])
                    in_position = True

            # Sell / Flat Signal (-1 or 0 when in_position)
            elif (sig == -1 or sig == 0) and in_position:
                adj_price = price * (1.0 - self.slippage)
                proceeds = (shares * adj_price) - self.commission
                cash += proceeds
                pnl = proceeds - (shares * entry_price)
                pnl_pct = ((adj_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
                exit_date = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(adj_price, 2),
                    "shares": round(shares, 4),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
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
            trades.append({
                "entry_date": entry_date,
                "exit_date": str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1]),
                "entry_price": round(entry_price, 2),
                "exit_price": round(last_price, 2),
                "shares": round(shares, 4),
                "pnl": round(unrealized_pnl, 2),
                "pnl_pct": round(unrealized_pct, 2),
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

        # Equity Curve Samples (~150 puntos para graficación rápida)
        step = max(1, len(pv_series) // 150)
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
        }
