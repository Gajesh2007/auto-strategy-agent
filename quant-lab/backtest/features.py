"""Feature engineering for ML trading strategies."""

import pandas as pd
import numpy as np
import ta


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator features to a DataFrame with OHLCV data."""
    df = df.copy()
    
    # Ensure we have the right column names
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # Price-based features
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        df[f"sma_{window}"] = close.rolling(window=window).mean()
        df[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()
    
    # SMA crossover signals
    df["sma_5_20_cross"] = (df["sma_5"] > df["sma_20"]).astype(int)
    df["sma_20_50_cross"] = (df["sma_20"] > df["sma_50"]).astype(int)
    df["sma_50_200_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)
    
    # Price relative to MAs
    df["price_to_sma_20"] = close / df["sma_20"]
    df["price_to_sma_50"] = close / df["sma_50"]
    df["price_to_sma_200"] = close / df["sma_200"]
    
    # Momentum indicators
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_7"] = ta.momentum.RSIIndicator(close, window=7).rsi()
    
    # MACD
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # Volatility
    for window in [5, 10, 20, 50]:
        df[f"volatility_{window}"] = df["returns"].rolling(window=window).std() * np.sqrt(252)
    
    # ATR
    df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"] = df["atr_14"] / close
    
    # Volume features
    df["volume_sma_20"] = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / df["volume_sma_20"]
    df["volume_std_20"] = volume.rolling(window=20).std()
    
    # Price momentum
    for period in [1, 5, 10, 20, 60]:
        df[f"momentum_{period}"] = close / close.shift(period) - 1
    
    # Rolling stats
    for window in [5, 10, 20]:
        df[f"rolling_max_{window}"] = high.rolling(window=window).max()
        df[f"rolling_min_{window}"] = low.rolling(window=window).min()
        df[f"distance_from_high_{window}"] = close / df[f"rolling_max_{window}"] - 1
        df[f"distance_from_low_{window}"] = close / df[f"rolling_min_{window}"] - 1
    
    # Trend features
    df["trend_5"] = np.where(close > close.shift(5), 1, -1)
    df["trend_20"] = np.where(close > close.shift(20), 1, -1)
    
    # Day of week (if datetime index)
    if hasattr(df.index, "dayofweek"):
        df["day_of_week"] = df.index.dayofweek
        df["is_monday"] = (df.index.dayofweek == 0).astype(int)
        df["is_friday"] = (df.index.dayofweek == 4).astype(int)
    
    return df


def get_feature_list(feature_set: str = "default") -> list[str]:
    """Get a predefined list of features."""
    
    base_momentum = [
        "sma_5", "sma_20", "sma_50",
        "sma_5_20_cross", "sma_20_50_cross",
        "price_to_sma_20", "price_to_sma_50",
        "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_diff",
        "momentum_1", "momentum_5", "momentum_20",
    ]
    
    volatility_features = [
        "volatility_10", "volatility_20",
        "bb_width", "bb_pct",
        "atr_pct",
    ]
    
    volume_features = [
        "volume_ratio",
    ]
    
    feature_sets = {
        "minimal": ["sma_5_20_cross", "rsi_14", "momentum_5"],
        "default": base_momentum + volatility_features[:3],
        "momentum": base_momentum,
        "full": base_momentum + volatility_features + volume_features,
        "volatility": volatility_features + ["rsi_14", "momentum_5"],
    }
    
    return feature_sets.get(feature_set, feature_sets["default"])

