import pandas as pd
import numpy as np


def generate_signals_multi_state(entry_conditions, exit_conditions, valid_mask, cooldowns=None, max_days=None):
    """
    Función utilitaria genérica que implementa una máquina de estados para generar 
    señales de trading (1, -1, 0) y tipos de señal activos de forma cronológica.
    
    entry_conditions: dict del tipo {nombre_estado: pd.Series (booleana)}
    exit_conditions: dict del tipo {nombre_estado: pd.Series (booleana)}
    valid_mask: pd.Series (booleana) indicando calentamiento de indicadores.
    cooldowns: dict opcional del tipo {nombre_estado: int} con min. velas fuera requeridas para reentrar.
    max_days: dict opcional del tipo {nombre_estado: int} con máx. velas dentro permitidas en posición.
    """
    keys = list(entry_conditions.keys())
    n = len(valid_mask)
    if cooldowns is None:
        cooldowns = {}
    if max_days is None:
        max_days = {}
    
    signals = np.zeros(n)
    sig_types = ['none'] * n
    active_state = 'none'
    bars_out = {k: 9999 for k in keys}
    bars_in = 0
    
    # Pre-extraer arrays de numpy para velocidad
    entry_arrs = {k: entry_conditions[k].values for k in keys}
    exit_arrs = {k: exit_conditions[k].values for k in keys}
    valid_arr = valid_mask.values
    
    for i in range(n):
        if not valid_arr[i]:
            signals[i] = 0
            sig_types[i] = 'none'
            active_state = 'none'
            bars_in = 0
            continue
            
        # Incrementar contador de velas fuera de posición y dentro de posición
        if active_state == 'none':
            for k in keys:
                bars_out[k] += 1
        else:
            bars_in += 1
            for k in keys:
                if k != active_state:
                    bars_out[k] += 1
            
        # 1. Evaluar salida de la posición actual
        if active_state != 'none':
            max_d = max_days.get(active_state, 999999)
            time_exit = bars_in >= max_d
            
            if exit_arrs[active_state][i] or time_exit:
                signals[i] = -1
                bars_out[active_state] = 0
                active_state = 'none'
                bars_in = 0
            # Transición directa: si estamos en meanreversion y se da la entrada de tendencia
            elif active_state == 'meanreversion' and 'trendfollowing' in entry_arrs and entry_arrs['trendfollowing'][i]:
                signals[i] = 0
                active_state = 'trendfollowing'
                bars_out['trendfollowing'] = 0
                bars_in = 0
            else:
                signals[i] = 0
                
        # 2. Evaluar entrada si no estamos posicionados (o acabamos de salir en la misma barra)
        if active_state == 'none':
            entered = False
            for k in keys:
                min_cd = cooldowns.get(k, 0)
                if entry_arrs[k][i] and bars_out[k] >= min_cd:
                    signals[i] = 1
                    active_state = k
                    bars_out[k] = 0
                    bars_in = 0
                    entered = True
                    break
            if not entered and signals[i] != -1:
                signals[i] = 0
                
        sig_types[i] = active_state
        
    return pd.Series(signals, index=valid_mask.index), pd.Series(sig_types, index=valid_mask.index)


def calculate_supertrend(df, lookback=28, multiplier=3.5):
    """
    Cálculo genérico de Supertrend indicador.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=lookback).mean()
    
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    
    b_upper = basic_upper.values
    b_lower = basic_lower.values
    c_vals = close.values
    
    for i in range(1, n):
        if np.isnan(b_upper[i]):
            final_upper[i] = c_vals[i]
            final_lower[i] = c_vals[i]
            supertrend[i] = c_vals[i]
            continue
            
        # Final Upper Band
        if (b_upper[i] < final_upper[i-1]) or (c_vals[i-1] > final_upper[i-1]):
            final_upper[i] = b_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # Final Lower Band
        if (b_lower[i] > final_lower[i-1]) or (c_vals[i-1] < final_lower[i-1]):
            final_lower[i] = b_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        # Supertrend
        if (supertrend[i-1] == final_upper[i-1]) and (c_vals[i] <= final_upper[i]):
            supertrend[i] = final_upper[i]
        elif (supertrend[i-1] == final_upper[i-1]) and (c_vals[i] > final_upper[i]):
            supertrend[i] = final_lower[i]
        elif (supertrend[i-1] == final_lower[i-1]) and (c_vals[i] >= final_lower[i]):
            supertrend[i] = final_lower[i]
        elif (supertrend[i-1] == final_lower[i-1]) and (c_vals[i] < final_lower[i]):
            supertrend[i] = final_upper[i]
        else:
            supertrend[i] = final_lower[i]
            
    return pd.DataFrame({'supertrend': supertrend}, index=df.index)


def lowest_low(data_df, params):
    """
    Estrategia de Reversión a la Media basada en Mínimo Más Bajo (Lowest Low).
    data_df: DataFrame con OHLCV
    Params: lookback, sl (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']
    low = data_df['Low']
    high = data_df['High']

    if isinstance(params, dict):
        lookback = params.get('lookback', 20)
        sl_val = params.get('sl', params.get('stop_loss', 5.0))
    elif isinstance(params, (list, tuple)):
        lookback = params[0] if len(params) > 0 else 20
        sl_val = params[1] if len(params) > 1 else 5.0
    else:
        lookback = 20
        sl_val = 5.0

    lowest_back = low.rolling(window=lookback).min()
    entry_cond = low == lowest_back
    exit_cond = close > high.shift(1)
    valid_mask = lowest_back.notnull()

    entry_conditions = {'meanreversion': entry_cond}
    exit_conditions = {'meanreversion': exit_cond}

    final_signals, signal_type = generate_signals_multi_state(entry_conditions, exit_conditions, valid_mask)

    return final_signals, pd.DataFrame({'Lowest_Back': lowest_back, 'SignalType': signal_type}, index=close.index)


def buy_the_zdip(data_df, params):
    """
    Estrategia de compra en z-dip.
    data_df: DataFrame con OHLCV
    Params: zwindow, zdema_period, zthreshold
    """
    data = data_df['Close']

    zwindow = params.get('zwindow', 10)
    zdema_period = params.get('zdema_period', 10)
    zthreshold = params.get('zthreshold', -2)

    zmean = data.rolling(window=zwindow).mean()
    zstd = data.rolling(window=zwindow).std()
    zscore = (data - zmean) / zstd

    ze1 = zscore.ewm(span=zdema_period, adjust=False).mean()
    ze2 = ze1.ewm(span=zdema_period, adjust=False).mean()
    zdema = 2 * ze1 - ze2

    high = data_df['High']
    long_mask = (zdema < zthreshold) & (zdema.shift() > zthreshold) & (zdema.notnull())
    exit_mask = (data > high.shift(1)) & (zdema.notnull())

    final_signals = np.where(long_mask, 1, 0)
    final_signals = np.where(exit_mask, -1, final_signals)
    
    return pd.Series(final_signals, index=data.index), pd.DataFrame({'ZScore': zscore, 'ZDEMA': zdema, 'SignalType': 'meanreversion'}, index=data.index)


def buy_the_zbounce(data_df, params):
    """
    Estrategia de compra en z-bounce.
    data_df: DataFrame con OHLCV
    Params: zwindow, zdema_period, zthreshold
    """
    data = data_df['Close']

    zwindow = params.get('zwindow', 10)
    zdema_period = params.get('zdema_period', 10)
    zthreshold = params.get('zthreshold', -2)

    zmean = data.rolling(window=zwindow).mean()
    zstd = data.rolling(window=zwindow).std()
    zscore = (data - zmean) / zstd

    ze1 = zscore.ewm(span=zdema_period, adjust=False).mean()
    ze2 = ze1.ewm(span=zdema_period, adjust=False).mean()
    zdema = 2 * ze1 - ze2

    high = data_df['High']
    long_mask = (zdema.shift() < zthreshold) & (zdema > zthreshold)
    exit_mask = (data > high.shift(1))

    final_signals = np.where(long_mask, 1, 0)
    final_signals = np.where(exit_mask, -1, final_signals)
    
    return pd.Series(final_signals, index=data.index), pd.DataFrame({'ZScore': zscore, 'ZDEMA': zdema, 'SignalType': 'meanreversion'}, index=data.index)
    

#MEAN REVERSION STRATEGIES
    
def rsi_weakness(data_df, params):
    """
    Estrategia de reversión a la media basada en debilidad del RSI.
    data_df: DataFrame con OHLCV
    Params: rsi_period, rsi_lower (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']
    high = data_df['High']

    if isinstance(params, dict):
        rsi_period = params.get('rsi_period', 14)
        rsi_lower = params.get('rsi_lower', 30)
    elif isinstance(params, (list, tuple)):
        rsi_period = params[0] if len(params) > 0 else 14
        rsi_lower = params[1] if len(params) > 1 else 30
    else:
        rsi_period = 14
        rsi_lower = 30

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    entry_cond = rsi < rsi_lower
    exit_cond = close > high.shift(1)
    valid_mask = rsi.notnull()

    entry_conditions = {'meanreversion': entry_cond}
    exit_conditions = {'meanreversion': exit_cond}

    final_signals, signal_type = generate_signals_multi_state(entry_conditions, exit_conditions, valid_mask)

    return final_signals, pd.DataFrame({'RSI': rsi, 'SignalType': signal_type}, index=close.index)


def bollinger_bands(data_df, params):
    """
    Estrategia de Reversión a la Media basada en Bandas de Bollinger.
    data_df: DataFrame con OHLCV
    Params: ma, mult (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']
    high = data_df['High']

    if isinstance(params, dict):
        ma = params.get('ma', 20)
        mult = params.get('mult', 2.0)
    elif isinstance(params, (list, tuple)):
        ma = params[0] if len(params) > 0 else 20
        mult = params[1] if len(params) > 1 else 2.0
    else:
        ma = 20
        mult = 2.0

    bb_mid = close.rolling(window=ma).mean()
    bb_std = close.rolling(window=ma).std()
    bb_down = bb_mid - (bb_std * mult)

    entry_cond = close < bb_down
    exit_cond = close > high.shift(1)
    valid_mask = bb_down.notnull()

    entry_conditions = {'meanreversion': entry_cond}
    exit_conditions = {'meanreversion': exit_cond}

    final_signals, signal_type = generate_signals_multi_state(entry_conditions, exit_conditions, valid_mask)

    return final_signals, pd.DataFrame({'BB_Down': bb_down, 'SignalType': signal_type}, index=close.index)


def money_flow_index(data_df, params):
    """
    Estrategia de Reversión a la Media basada en Money Flow Index (MFI).
    data_df: DataFrame con OHLCV
    Params: mfi_periods, in_mfi, days_in (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']
    high = data_df['High']
    low = data_df['Low']
    volume = data_df['Volume'] if 'Volume' in data_df.columns else pd.Series(1, index=close.index)

    if isinstance(params, dict):
        mfi_periods = params.get('mfi_periods', 14)
        in_mfi = params.get('in_mfi', 20)
        days_in = params.get('days_in', 5)
    elif isinstance(params, (list, tuple)):
        mfi_periods = params[0] if len(params) > 0 else 14
        in_mfi = params[1] if len(params) > 1 else 20
        days_in = params[2] if len(params) > 2 else 5
    else:
        mfi_periods = 14
        in_mfi = 20
        days_in = 5

    tp = (high + low + close) / 3
    rmf = tp * volume
    close_diff = close.diff()
    pmf = rmf.where(close_diff > 0, 0)
    nmf = rmf.where(close_diff < 0, 0)

    pmf_sum = pmf.rolling(window=mfi_periods).sum()
    nmf_sum = nmf.rolling(window=mfi_periods).sum()

    money_ratio = pmf_sum / nmf_sum
    mfi = 100 - (100 / (1 + money_ratio))

    entry_cond = mfi < in_mfi
    exit_cond = close > high.shift(1)
    valid_mask = mfi.notnull()

    entry_conditions = {'meanreversion': entry_cond}
    exit_conditions = {'meanreversion': exit_cond}
    max_days = {'meanreversion': days_in}

    final_signals, signal_type = generate_signals_multi_state(
        entry_conditions, exit_conditions, valid_mask, max_days=max_days
    )

    return final_signals, pd.DataFrame({'MFI': mfi, 'SignalType': signal_type}, index=close.index)


def regression_rsi_short(data_df, params):
    """
    Estrategia de Reversión a la Media basada en Regresión Lineal y RSI.
    data_df: DataFrame con OHLCV
    Params: day_reg/regression_period, rsi_win/rsi_period (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']
    low = data_df['Low']

    if isinstance(params, dict):
        day_reg = params.get('day_reg', params.get('regression_period', 10))
        rsi_win = params.get('rsi_win', params.get('rsi_period', 14))
        rsi_upper = params.get('rsi_upper', 70)
    elif isinstance(params, (list, tuple)):
        day_reg = params[0] if len(params) > 0 else 10
        rsi_win = params[1] if len(params) > 1 else 14
        rsi_upper = params[2] if len(params) > 2 else 70
    else:
        day_reg = 10
        rsi_win = 14
        rsi_upper = 70

    # Pendiente de regresión lineal móvil
    y = close.values
    window = day_reg
    x = np.arange(window)
    sum_x, sum_x2 = np.sum(x), np.sum(x**2)
    divisor = window * sum_x2 - sum_x**2
    slopes = [np.nan] * (window - 1)
    for i in range(window, len(y) + 1):
        y_slice = y[i-window:i]
        m = (window * np.sum(x * y_slice) - sum_x * np.sum(y_slice)) / divisor
        slopes.append(m)
    slope_series = pd.Series(slopes, index=close.index)

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_win).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_win).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    entry_cond = (slope_series < 0) & (rsi > rsi_upper)
    exit_cond = close < low.shift(1)
    valid_mask = slope_series.notnull() & rsi.notnull()

    entry_conditions = {'meanreversion': entry_cond}
    exit_conditions = {'meanreversion': exit_cond}

    final_signals, signal_type = generate_signals_multi_state(entry_conditions, exit_conditions, valid_mask)

    return final_signals, pd.DataFrame({'Slope': slope_series, 'RSI': rsi, 'SignalType': signal_type}, index=close.index)

#TREND FOLLOWING STRATEGIES

def golden_cross(data_df, params):
    """
    Estrategia de Seguimiento de Tendencia basada en Cruce Dorado de EMAs y Supertrend.
    data_df: DataFrame con OHLCV
    Params: ema1/window_rapid, ema2/window_slow, cooldown (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']

    if isinstance(params, dict):
        ema1 = params.get('ema1', params.get('window_rapid', 50))
        ema2 = params.get('ema2', params.get('window_slow', 200))
        cooldown = params.get('cooldown', 16)
    elif isinstance(params, (list, tuple)):
        ema1 = params[0] if len(params) > 0 else 50
        ema2 = params[1] if len(params) > 1 else 200
        cooldown = params[2] if len(params) > 2 else 16
    else:
        ema1 = 50
        ema2 = 200
        cooldown = 16

    ema1_series = close.ewm(span=ema1, adjust=False).mean()
    ema2_series = close.ewm(span=ema2, adjust=False).mean()

    # Supertrend
    st_df = calculate_supertrend(data_df, lookback=28, multiplier=3.5)
    supertrend_val = st_df['supertrend']
    sup_bul_bear = np.where(supertrend_val > close, -1, 1)
    sp_change = pd.Series(sup_bul_bear, index=close.index) != pd.Series(sup_bul_bear, index=close.index).shift(1)

    entry_cond = ema1_series > ema2_series
    exit_ema = ema1_series < ema2_series
    exit_sp = (close < supertrend_val) & sp_change
    exit_cond = exit_ema | exit_sp

    entry_conditions = {'trendfollowing': entry_cond}
    exit_conditions = {'trendfollowing': exit_cond}
    cooldowns = {'trendfollowing': cooldown}

    valid_mask = ema1_series.notnull() & ema2_series.notnull() & supertrend_val.notnull()

    final_signals, signal_type = generate_signals_multi_state(
        entry_conditions, exit_conditions, valid_mask, cooldowns=cooldowns
    )

    return final_signals, pd.DataFrame({
        'EMA1': ema1_series,
        'EMA2': ema2_series,
        'Supertrend': supertrend_val,
        'SignalType': signal_type
    }, index=close.index)


def kvo_bull_spt(data_df, params):
    """
    Estrategia de Seguimiento de Tendencia: KVO + EMA + Supertrend.
    data_df: DataFrame con OHLCV
    Params: kvo ([fast, slow]), ema (span), cooldown (velas) (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']
    high = data_df['High']
    low = data_df['Low']
    volume = data_df['Volume'] if 'Volume' in data_df.columns else pd.Series(1, index=close.index)

    if isinstance(params, dict):
        kvo_params = params.get('kvo', [7, 28])
        ema_span = params.get('ema', 200)
        cooldown = params.get('cooldown', 12)
    elif isinstance(params, (list, tuple)):
        kvo_params = params[0] if len(params) > 0 else [7, 28]
        ema_span = params[1] if len(params) > 1 else 200
        cooldown = params[2] if len(params) > 2 else 12
    else:
        kvo_params = [7, 28]
        ema_span = 200
        cooldown = 12

    fastT = kvo_params[0] if isinstance(kvo_params, (list, tuple)) else 7
    slowT = kvo_params[1] if isinstance(kvo_params, (list, tuple)) else 28

    # Indicador KVO
    hlc3 = (high + low + close) / 3
    kvo_trend = np.where(hlc3 > hlc3.shift(1), volume * 100, -volume * 100)
    kvo_fast = pd.Series(kvo_trend, index=close.index).ewm(span=fastT, adjust=False).mean()
    kvo_slow = pd.Series(kvo_trend, index=close.index).ewm(span=slowT, adjust=False).mean()
    kvo = kvo_fast - kvo_slow

    # EMAs
    ema_0 = close.ewm(span=5, adjust=False).mean()
    ema_main = close.ewm(span=ema_span, adjust=False).mean()

    # Supertrend
    st_df = calculate_supertrend(data_df, lookback=28, multiplier=3.5)
    supertrend_val = st_df['supertrend']
    sup_bul_bear = np.where(supertrend_val > close, -1, 1)
    sp_change = pd.Series(sup_bul_bear, index=close.index) != pd.Series(sup_bul_bear, index=close.index).shift(1)

    # Condiciones
    bullish_cond = ema_0 > ema_main
    kvo_cross_up = (kvo > 0) & (kvo.shift(1) < 0)
    cond_entry = kvo_cross_up & bullish_cond

    exit_ema = (close > high.shift(1)) & (ema_0 < ema_main)
    exit_sp = (close < supertrend_val) & sp_change
    cond_exit = exit_ema | exit_sp

    entry_conditions = {'trendfollowing': cond_entry}
    exit_conditions = {'trendfollowing': cond_exit}
    cooldowns = {'trendfollowing': cooldown}

    valid_mask = kvo.notnull() & ema_main.notnull() & supertrend_val.notnull()

    final_signals, signal_type = generate_signals_multi_state(
        entry_conditions, exit_conditions, valid_mask, cooldowns=cooldowns
    )

    return final_signals, pd.DataFrame({
        'KVO': kvo,
        'EMA_0': ema_0,
        'EMA': ema_main,
        'Supertrend': supertrend_val,
        'SignalType': signal_type
    }, index=close.index)


def macd_slope(data_df, params):
    """
    Estrategia de Seguimiento de Tendencia basada en Cruce de MACD y Filtro de EMA.
    data_df: DataFrame con OHLCV
    Params: macd ([fast, slow, signal]), ema (span) (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']

    if isinstance(params, dict):
        macd_inputs = params.get('macd', [12, 26, 9])
        ema_span = params.get('ema', 200)
    elif isinstance(params, (list, tuple)):
        macd_inputs = params[0] if len(params) > 0 else [12, 26, 9]
        ema_span = params[1] if len(params) > 1 else 200
    else:
        macd_inputs = [12, 26, 9]
        ema_span = 200

    fast_period = macd_inputs[0] if isinstance(macd_inputs, (list, tuple)) else 12
    slow_period = macd_inputs[1] if isinstance(macd_inputs, (list, tuple)) else 26
    signal_period = macd_inputs[2] if isinstance(macd_inputs, (list, tuple)) else 9

    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    ema_0 = close.ewm(span=5, adjust=False).mean()
    ema_main = close.ewm(span=ema_span, adjust=False).mean()

    bullish = ema_0 > ema_main
    entry_cond = (macd_line > 0) & (macd_line.shift(1) < 0) & bullish
    exit_cond = (macd_line < 0) | (~bullish)

    valid_mask = macd_line.notnull() & signal_line.notnull() & ema_main.notnull()

    entry_conditions = {'trendfollowing': entry_cond}
    exit_conditions = {'trendfollowing': exit_cond}

    final_signals, signal_type = generate_signals_multi_state(entry_conditions, exit_conditions, valid_mask)

    return final_signals, pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'EMA_0': ema_0,
        'EMA': ema_main,
        'SignalType': signal_type
    }, index=close.index)


def macd_slope_spt(data_df, params):
    """
    Estrategia de Seguimiento de Tendencia: MACD + EMA + Supertrend.
    data_df: DataFrame con OHLCV
    Params: macd ([fast, slow, signal]), ema (span), cooldown (velas) (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']

    if isinstance(params, dict):
        macd_inputs = params.get('macd', [12, 26, 9])
        ema_span = params.get('ema', 200)
        cooldown = params.get('cooldown', 21)
    elif isinstance(params, (list, tuple)):
        macd_inputs = params[0] if len(params) > 0 else [12, 26, 9]
        ema_span = params[1] if len(params) > 1 else 200
        cooldown = params[2] if len(params) > 2 else 21
    else:
        macd_inputs = [12, 26, 9]
        ema_span = 200
        cooldown = 21

    fast_period = macd_inputs[0] if isinstance(macd_inputs, (list, tuple)) else 12
    slow_period = macd_inputs[1] if isinstance(macd_inputs, (list, tuple)) else 26
    signal_period = macd_inputs[2] if isinstance(macd_inputs, (list, tuple)) else 9

    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    ema_0 = close.ewm(span=5, adjust=False).mean()
    ema_main = close.ewm(span=ema_span, adjust=False).mean()

    # Supertrend
    st_df = calculate_supertrend(data_df, lookback=28, multiplier=4.0)
    supertrend_val = st_df['supertrend']
    sup_bul_bear = np.where(supertrend_val > close, -1, 1)
    sp_change = pd.Series(sup_bul_bear, index=close.index) != pd.Series(sup_bul_bear, index=close.index).shift(1)

    bullish = ema_0 > ema_main
    entry_cond = (macd_line > 0) & (macd_line.shift(1) < 0) & bullish
    exit_macd = (macd_line < 0) | (~bullish)
    exit_sp = (close < supertrend_val) & sp_change
    exit_cond = exit_macd | exit_sp

    entry_conditions = {'trendfollowing': entry_cond}
    exit_conditions = {'trendfollowing': exit_cond}
    cooldowns = {'trendfollowing': cooldown}

    valid_mask = macd_line.notnull() & signal_line.notnull() & ema_main.notnull() & supertrend_val.notnull()

    final_signals, signal_type = generate_signals_multi_state(
        entry_conditions, exit_conditions, valid_mask, cooldowns=cooldowns
    )

    return final_signals, pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'EMA_0': ema_0,
        'EMA': ema_main,
        'Supertrend': supertrend_val,
        'SignalType': signal_type
    }, index=close.index)
   

def trend_pull_back_rsi(data_df, params):
    """
    Estrategia de Reversión a la Media en Tendencia usando RSI, MAs Pocket y Supertrend.
    data_df: DataFrame con OHLCV
    Params: rsi_window, out_rsi, in_rsi, ma_slow, ma_fast, cooldown (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']

    if isinstance(params, dict):
        rsi_window = params.get('rsi_window', params.get('rsi_period', 14))
        out_rsi = params.get('out_rsi', 70)
        in_rsi = params.get('in_rsi', params.get('rsi_lower', 30))
        ma_slow = params.get('ma_slow', 200)
        ma_fast = params.get('ma_fast', 50)
        cooldown = params.get('cooldown', 7)
    elif isinstance(params, (list, tuple)):
        rsi_window = params[0] if len(params) > 0 else 14
        out_rsi = params[1] if len(params) > 1 else 70
        in_rsi = params[2] if len(params) > 2 else 30
        ma_slow = params[3] if len(params) > 3 else 200
        ma_fast = params[4] if len(params) > 4 else 50
        cooldown = params[5] if len(params) > 5 else 7
    else:
        rsi_window = 14
        out_rsi = 70
        in_rsi = 30
        ma_slow = 200
        ma_fast = 50
        cooldown = 7

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MAs
    ma_s = close.rolling(window=ma_slow).mean()
    ma_f = close.rolling(window=ma_fast).mean()
    ma_pocket = (close < ma_f) & (close > ma_s)

    # Supertrend
    st_df = calculate_supertrend(data_df, lookback=56, multiplier=3.5)
    supertrend_val = st_df['supertrend']
    sup_bul_bear = np.where(supertrend_val > close, -1, 1)
    sp_change = pd.Series(sup_bul_bear, index=close.index) != pd.Series(sup_bul_bear, index=close.index).shift(1)

    entry_cond = ma_pocket & (rsi < in_rsi)
    exit_cond = (rsi > out_rsi) | ((close < supertrend_val) & sp_change)

    entry_conditions = {'meanreversion': entry_cond}
    exit_conditions = {'meanreversion': exit_cond}
    cooldowns = {'meanreversion': cooldown}

    valid_mask = rsi.notnull() & ma_s.notnull() & supertrend_val.notnull()

    final_signals, signal_type = generate_signals_multi_state(
        entry_conditions, exit_conditions, valid_mask, cooldowns=cooldowns
    )

    return final_signals, pd.DataFrame({
        'RSI': rsi,
        'MA_Slow': ma_s,
        'MA_Fast': ma_f,
        'Supertrend': supertrend_val,
        'SignalType': signal_type
    }, index=close.index)


def buy_weakness(data_df, params):
    """
    Estrategia de Reversión a la Media basada en Puntuación de Debilidad (Multi-Indicador).
    data_df: DataFrame con OHLCV
    Params: macd ([fast, slow, signal]), weak_in, wr (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']
    high = data_df['High']
    low = data_df['Low']

    if isinstance(params, dict):
        macd_inputs = params.get('macd', [12, 26, 9])
        weak_in = params.get('weak_in', 5)
        wr_period = params.get('wr', 14)
    elif isinstance(params, (list, tuple)):
        macd_inputs = params[0] if len(params) > 0 else [12, 26, 9]
        weak_in = params[1] if len(params) > 1 else 5
        wr_period = params[2] if len(params) > 2 else 14
    else:
        macd_inputs = [12, 26, 9]
        weak_in = 5
        wr_period = 14

    fast_period = macd_inputs[0] if isinstance(macd_inputs, (list, tuple)) else 12
    slow_period = macd_inputs[1] if isinstance(macd_inputs, (list, tuple)) else 26
    signal_period = macd_inputs[2] if isinstance(macd_inputs, (list, tuple)) else 9

    # DEMA 5
    e1 = close.ewm(span=5, adjust=False).mean()
    e2 = e1.ewm(span=5, adjust=False).mean()
    dema5 = 2 * e1 - e2

    # EMAs
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # MACD
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # Pendientes móviles (ventana de 3)
    macd_slope = (macd_line - macd_line.shift(2)) / 2
    hist_slope = (macd_hist - macd_hist.shift(2)) / 2

    # Supertrend
    st_df = calculate_supertrend(data_df, lookback=fast_period, multiplier=2.0)
    supertrend_val = st_df['supertrend']

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Williams %R
    highest_high = high.rolling(window=wr_period).max()
    lowest_low = low.rolling(window=wr_period).min()
    wr_indicator = ((highest_high - close) / (highest_high - lowest_low)) * -100

    # Condiciones de puntuación
    c_ema50 = np.where(dema5 > ema50, 1, -1)
    c_ema100 = np.where(ema100 < ema50, 1, -1)
    c_ema200 = np.where(ema200 < ema50, 1, -1)
    c_macdline = np.where(macd_line > 0, 1, -1)
    c_macdgap = np.where(macd_line > signal_line, 1, -1)
    c_macdslope = np.where(macd_slope > 0, 1, -1)
    c_macdsbarras = np.where(hist_slope > 0, 1, -1)
    c_supert = np.where(close > supertrend_val, 1, -1)
    c_rsi = np.where(rsi > 70, 1, np.where(rsi < 30, -1, 0))
    c_wr = np.where(wr_indicator > -30, 1, np.where(wr_indicator < -30, -1, 0))

    score_sum = pd.Series(
        c_ema50 + c_ema100 + c_ema200 + c_macdline + c_macdgap + c_macdslope + c_macdsbarras + c_supert + c_rsi + c_wr,
        index=close.index
    )

    entry_cond = (score_sum == weak_in) & (score_sum.shift(1) > weak_in)
    exit_cond = close > high.shift(1)
    valid_mask = dema5.notnull() & ema200.notnull() & macd_line.notnull() & wr_indicator.notnull()

    entry_conditions = {'meanreversion': entry_cond}
    exit_conditions = {'meanreversion': exit_cond}

    final_signals, signal_type = generate_signals_multi_state(entry_conditions, exit_conditions, valid_mask)

    return final_signals, pd.DataFrame({'ScoreSum': score_sum, 'SignalType': signal_type}, index=close.index)
  
# MOMENTUM


def zscore_bull(data_df, params):
    """
    Estrategia de Seguimiento de Tendencia/Momento basada en Z-Score y EMA.
    data_df: DataFrame con OHLCV
    Params: zscore_period, z_in, days_in, ema (acepta diccionario o lista/tupla)
    """
    close = data_df['Close']

    if isinstance(params, dict):
        zscore_period = params.get('zscore_period', 20)
        z_in = params.get('z_in', 1.0)
        days_in = params.get('days_in', 10)
        ema_span = params.get('ema', 200)
    elif isinstance(params, (list, tuple)):
        zscore_period = params[0] if len(params) > 0 else 20
        z_in = params[1] if len(params) > 1 else 1.0
        days_in = params[2] if len(params) > 2 else 10
        ema_span = params[3] if len(params) > 3 else 200
    else:
        zscore_period = 20
        z_in = 1.0
        days_in = 10
        ema_span = 200

    # Z-Score
    z_mean = close.rolling(window=zscore_period).mean()
    z_std = close.rolling(window=zscore_period).std()
    zscore = (close - z_mean) / z_std

    # EMAs
    ema_0 = close.ewm(span=5, adjust=False).mean()
    ema_main = close.ewm(span=ema_span, adjust=False).mean()

    bullish = ema_0 > ema_main
    entry_cond = (zscore > z_in) & (zscore.shift(1) < z_in) & bullish

    cond_out = ((zscore < 2) & (zscore.shift(1) > 2)) | ((zscore < 1) & (zscore.shift(1) > 1))
    exit_cond = cond_out | (~bullish)

    valid_mask = zscore.notnull() & ema_main.notnull()

    entry_conditions = {'trendfollowing': entry_cond}
    exit_conditions = {'trendfollowing': exit_cond}
    max_days = {'trendfollowing': days_in}

    final_signals, signal_type = generate_signals_multi_state(
        entry_conditions, exit_conditions, valid_mask, max_days=max_days
    )

    return final_signals, pd.DataFrame({
        'ZScore': zscore,
        'EMA_0': ema_0,
        'EMA': ema_main,
        'SignalType': signal_type
    }, index=close.index)



#COMBINED STRATEGIES

def zs_cross_dema_combox2(data_df, params):
    """
    Estrategia combinada optimizada: Cruce de DEMA + Z-Score + RSI + Pendiente.
    data_df: DataFrame con OHLCV
    Params: window_rapid, window_slow, zscore_period, zdema_period, rsi_period, rsi_lower
    """
    # 0. Preparar series
    close = data_df['Close']
    high = data_df['High']
    
    # Parámetros (default coincidentes con test_model_4.py)
    w_rapid = params.get('window_rapid', 14)
    w_slow = params.get('window_slow', 90)
    z_period = params.get('zscore_period', 112)
    zd_period = params.get('zdema_period', 28)
    rsi_p = params.get('rsi_period', 3)
    rsi_low = params.get('rsi_lower', 30)

    # 1. DEMA Rapid/Slow (adjust=True coincide exactamente con test_model_4.py)
    e1_r = close.ewm(span=w_rapid, adjust=True).mean()
    e2_r = e1_r.ewm(span=w_rapid, adjust=True).mean()
    d_rapid = 2 * e1_r - e2_r

    e1_s = close.ewm(span=w_slow, adjust=True).mean()
    e2_s = e1_s.ewm(span=w_slow, adjust=True).mean()
    d_slow = 2 * e1_s - e2_s
    
    # 2. Z-Score y ZDEMA (calculado sobre el Z-Score, NO sobre el precio Close)
    z_mean = close.rolling(window=z_period).mean()
    z_std = close.rolling(window=z_period).std()
    zscore = (close - z_mean) / z_std
    
    ze1 = zscore.ewm(span=zd_period, adjust=True).mean()
    ze2 = ze1.ewm(span=zd_period, adjust=True).mean()
    zdema = 2 * ze1 - ze2

    # Pendiente de ZDEMA sobre ventana de 3 velas (equivalente exacto a polyfit(deg=1)): (zdema_t - zdema_{t-2})/2
    zdema_slope = (zdema - zdema.shift(2)) / 2
    
    # 3. RSI
    change = close.diff()
    change_up = np.where(change > 0, change, 0)
    change_down = np.where(change < 0, np.abs(change), 0)
    avg_up = pd.Series(change_up, index=close.index).rolling(rsi_p).mean()
    avg_down = pd.Series(change_down, index=close.index).rolling(rsi_p).mean()
    rsi = (100 * avg_up) / (avg_up + avg_down)

    # Lógica de Compra (Cond1 o Cond2)
    cond1 = ((d_rapid > d_slow) & (zdema > 1)) | ((d_rapid > d_slow) & (zdema < 1) & (zdema_slope > 0))
    cond2 = (~cond1) & (rsi < rsi_low)

    # Condición de salida de reversión a la media (Close > High.shift(1))
    exit_mr = close > high.shift(1)

    entry_conditions = {
        'trendfollowing': cond1,
        'meanreversion': cond2
    }

    exit_conditions = {
        'trendfollowing': ~cond1,
        'meanreversion': exit_mr
    }

    # En el período inicial de calentamiento de indicadores (NaNs), mantener sin posición (0)
    valid_mask = d_slow.notnull() & zdema.notnull() & zdema_slope.notnull() & rsi.notnull()

    final_signals, signal_type = generate_signals_multi_state(entry_conditions, exit_conditions, valid_mask)

    return final_signals, pd.DataFrame({'ZScore': zscore, 'zDEMA': zdema, 'zdema_slope': zdema_slope, 'rsi': rsi, 'SignalType': signal_type}, index=close.index)



# Diccionario de registro para fácil acceso
STRATEGY_MAP = {
    'MeanRev_WeakRSI': rsi_weakness,
    'MeanRev_BollingerBands': bollinger_bands,
    'bollinger_bands': bollinger_bands,
    'MeanRev_MFI': money_flow_index,
    'money_flow_index': money_flow_index,
    'MeanRev_PullBackRSI': trend_pull_back_rsi,
    'trend_pull_back_rsi': trend_pull_back_rsi,
    'MeanRev_BuyWeakness': buy_weakness,
    'buy_weakness': buy_weakness,
    'MeanRev_RegresRSIS': regression_rsi_short,
    'regression_rsi_short': regression_rsi_short,
    'MeanRev_LowestLow': lowest_low,
    'lowest_low': lowest_low,
    'TrendFollowing_GoldCross': golden_cross,
    'golden_cross': golden_cross,
    'TrendFollowing_MACDSlope': macd_slope,
    'macd_slope': macd_slope,
    'TrendFollowing_MACDSlopeSPT': macd_slope_spt,
    'macd_slope_spt': macd_slope_spt,
    'TrendFollowing_KVOBullSPT': kvo_bull_spt,
    'TrendFollowing_ZScoreBull': zscore_bull,
    'Momentum_ZscoreBull': zscore_bull,
    'zscore_bull': zscore_bull,
    'Combo_ZCrossDema': zs_cross_dema_combox2,
    'buy_the_zdip': buy_the_zdip,
    'buy_the_zbounce': buy_the_zbounce,
}
