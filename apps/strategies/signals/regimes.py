import pandas as pd


def compute_macro_regimes(
    hyg_closes: pd.Series,
    lqd_closes: pd.Series,
    dxy_closes: pd.Series,
    vix_closes: pd.Series,
    zscore_window: int = 48,
    slope_window: int = 3,
    vix_threshold: float = 20.0,
) -> pd.DataFrame:
    """Compatibilidad: la lógica vive en `apps.strategies.regimes.macro` (proveedor `macro`)."""
    # Import diferido: evita el ciclo signals -> regimes.macro -> signals.indicators
    from apps.strategies.regimes.macro import macro_regime_frame
    return macro_regime_frame(
        hyg_closes, lqd_closes, dxy_closes, vix_closes,
        zscore_window=zscore_window, slope_window=slope_window, vix_threshold=vix_threshold,
    )
