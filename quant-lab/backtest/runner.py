"""Backtest runner - orchestrates data loading, training, and backtesting."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .features import add_all_features, get_feature_list
from .models import get_model, train_model, predict_signals, get_feature_importance


def load_market_data(
    ticker: str,
    start_date: str,
    end_date: str,
    data_dir: str = "../data",
) -> pd.DataFrame:
    """Load market data, checking local cache first."""
    
    data_path = Path(data_dir)
    
    # Try to find cached data
    for file in data_path.glob(f"{ticker}_*.json"):
        with open(file) as f:
            cached = json.load(f)
            if "data" in cached:
                df = pd.DataFrame(cached["data"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("timestamp")
                
                # Check if date range is covered
                df_start = df.index.min().strftime("%Y-%m-%d")
                df_end = df.index.max().strftime("%Y-%m-%d")
                
                if df_start <= start_date and df_end >= end_date:
                    # Filter to requested range
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    return df[mask]
    
    # Fall back to yfinance
    print(f"Fetching {ticker} from yfinance...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df.columns = [c.lower() for c in df.columns]
    
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df


def backtest_signals(
    df: pd.DataFrame,
    signals: np.ndarray,
    transaction_cost: float = 0.001,
) -> dict[str, Any]:
    """Backtest trading signals and compute performance metrics."""
    
    df = df.copy()
    df["signal"] = signals
    
    # Next period returns
    df["forward_return"] = df["close"].pct_change().shift(-1)
    
    # Strategy returns (signal today, return tomorrow)
    df["strategy_return"] = df["signal"].shift(1) * df["forward_return"]
    
    # Account for transaction costs on position changes
    df["position_change"] = df["signal"].diff().abs()
    df["costs"] = df["position_change"] * transaction_cost
    df["strategy_return_net"] = df["strategy_return"] - df["costs"]
    
    # Drop NaN
    df = df.dropna()
    
    if len(df) < 10:
        return {
            "sharpe": 0.0,
            "totalReturn": 0.0,
            "maxDrawdown": 1.0,
            "winRate": 0.0,
            "profitFactor": 0.0,
            "numTrades": 0,
        }
    
    # Equity curve
    df["equity"] = (1 + df["strategy_return_net"]).cumprod()
    equity_curve = df["equity"].tolist()
    
    # Metrics
    returns = df["strategy_return_net"]
    
    # Sharpe ratio (annualized)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Total return
    total_return = df["equity"].iloc[-1] - 1
    
    # Max drawdown
    rolling_max = df["equity"].cummax()
    drawdowns = (df["equity"] - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min())
    
    # Win rate
    trades = df[df["signal"].shift(1) != 0]["strategy_return_net"]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0.0
    
    # Profit factor
    gains = trades[trades > 0].sum()
    losses = abs(trades[trades < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf")
    
    # Number of trades (position changes)
    num_trades = int(df["position_change"].sum() / 2)  # Divide by 2 because entry+exit = 2 changes
    
    return {
        "sharpe": float(sharpe),
        "totalReturn": float(total_return),
        "maxDrawdown": float(max_drawdown),
        "winRate": float(win_rate),
        "profitFactor": float(min(profit_factor, 100)),  # Cap at 100
        "numTrades": num_trades,
        "equityCurve": equity_curve,
    }


def run_backtest(
    ticker: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    model_type: str = "xgboost",
    horizon: int = 1,
    features: list[str] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    data_dir: str = "../data",
) -> dict[str, Any]:
    """Run a complete backtest pipeline."""
    
    # Load data for full range
    full_start = min(train_start, test_start)
    full_end = max(train_end, test_end)
    
    df = load_market_data(ticker, full_start, full_end, data_dir)
    
    if len(df) < 100:
        raise ValueError(f"Insufficient data: only {len(df)} rows")
    
    # Add features
    df = add_all_features(df)
    
    # Get feature list
    if features is None:
        features = get_feature_list("default")
    
    # Filter to features that exist
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 3:
        raise ValueError(f"Not enough features available. Requested: {features}, Available: {list(df.columns)}")
    
    # Create target: sign of return N days ahead
    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)
    
    # Drop NaN
    df = df.dropna(subset=available_features + ["target"])
    
    # Split data
    train_mask = (df.index >= train_start) & (df.index <= train_end)
    test_mask = (df.index >= test_start) & (df.index <= test_end)
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    if len(train_df) < 50:
        raise ValueError(f"Insufficient training data: {len(train_df)} rows")
    if len(test_df) < 20:
        raise ValueError(f"Insufficient test data: {len(test_df)} rows")
    
    # Prepare arrays
    X_train = train_df[available_features].values
    y_train = train_df["target"].values
    X_test = test_df[available_features].values
    
    # Get and train model
    model = get_model(model_type, hyperparameters)
    train_metrics = train_model(model, X_train, y_train, available_features)
    
    # Generate signals
    signals = predict_signals(model, X_test)
    
    # Backtest
    backtest_metrics = backtest_signals(test_df, signals)
    
    # Feature importance
    feature_importance = get_feature_importance(model, available_features)
    
    # Compile result
    result = {
        "id": str(uuid.uuid4())[:8],
        "config": {
            "ticker": ticker,
            "trainStart": train_start,
            "trainEnd": train_end,
            "testStart": test_start,
            "testEnd": test_end,
            "modelType": model_type,
            "horizon": horizon,
            "features": available_features,
            "hyperparameters": hyperparameters or {},
        },
        "metrics": {
            "sharpe": backtest_metrics["sharpe"],
            "totalReturn": backtest_metrics["totalReturn"],
            "maxDrawdown": backtest_metrics["maxDrawdown"],
            "winRate": backtest_metrics["winRate"],
            "profitFactor": backtest_metrics["profitFactor"],
            "numTrades": backtest_metrics["numTrades"],
        },
        "trainMetrics": train_metrics,
        "featureImportance": feature_importance,
        "equityCurve": backtest_metrics.get("equityCurve", []),
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    return result

