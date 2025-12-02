"""ML models for trading signal generation."""

from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb


def get_model(model_type: str, hyperparameters: dict[str, Any] | None = None):
    """Get a model instance based on type."""
    
    params = hyperparameters or {}
    
    if model_type == "logistic":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            class_weight=params.get("class_weight", "balanced"),
            random_state=42,
        )
    
    elif model_type == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            scale_pos_weight=params.get("scale_pos_weight", 1),
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 5),
            class_weight=params.get("class_weight", "balanced"),
            random_state=42,
            n_jobs=-1,
        )
    
    elif model_type == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            class_weight=params.get("class_weight", "balanced"),
            random_state=42,
            verbose=-1,
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Train a model and return training metrics."""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Store scaler on model for later use
    model._scaler = scaler
    model._feature_names = feature_names
    
    # Training predictions
    y_pred = model.predict(X_train_scaled)
    
    return {
        "accuracy": float(accuracy_score(y_train, y_pred)),
        "precision": float(precision_score(y_train, y_pred, zero_division=0)),
        "recall": float(recall_score(y_train, y_pred, zero_division=0)),
        "f1": float(f1_score(y_train, y_pred, zero_division=0)),
    }


def predict_signals(model, X_test: np.ndarray) -> np.ndarray:
    """Generate trading signals from a trained model."""
    
    if hasattr(model, "_scaler"):
        X_test_scaled = model._scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_scaled)
        
        # Convert to signals: -1 (short), 0 (no position), 1 (long)
        # Using probability thresholds
        signals = np.zeros(len(probs))
        signals[probs[:, 1] > 0.55] = 1  # Long if > 55% prob of up
        signals[probs[:, 1] < 0.45] = -1  # Short if < 45% prob of up
        
        return signals
    else:
        # Binary prediction to signal
        preds = model.predict(X_test_scaled)
        return np.where(preds == 1, 1, -1)


def get_feature_importance(model, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importance from a trained model."""
    
    importance = {}
    
    if hasattr(model, "feature_importances_"):
        # Tree-based models
        importances = model.feature_importances_
        total = sum(importances)
        for name, imp in zip(feature_names, importances):
            importance[name] = float(imp / total) if total > 0 else 0.0
    
    elif hasattr(model, "coef_"):
        # Linear models
        coefs = np.abs(model.coef_[0])
        total = sum(coefs)
        for name, coef in zip(feature_names, coefs):
            importance[name] = float(coef / total) if total > 0 else 0.0
    
    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    return importance

