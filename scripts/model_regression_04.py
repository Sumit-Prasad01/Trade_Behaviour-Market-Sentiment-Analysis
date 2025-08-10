import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

# ==============================
# 0. BASELINE REGRESSORS
# ==============================
def train_regressors(X_train, y_train):
    """Train baseline regressors."""
    # Linear Regression
    pipe_lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])
    pipe_lr.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # XGBoost Regressor
    xgb_r = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
    xgb_r.fit(X_train, y_train)

    return {'linreg': pipe_lr, 'rf': rf, 'xgb_reg': xgb_r}

# ==============================
# 1. HYPERPARAMETER TUNING
# ==============================
def tune_xgb_regressor(X_train, y_train):
    """Hyperparameter tuning for XGBRegressor."""
    tscv = TimeSeriesSplit(n_splits=5)
    xgb = XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    grid = GridSearchCV(
        xgb,
        param_grid,
        scoring='neg_mean_squared_error',
        cv=tscv,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print("Best Params (Regression):", grid.best_params_)
    return grid.best_estimator_

# ==============================
# 2. FEATURE IMPORTANCE (SHAP)
# ==============================
def plot_shap_importance(model, X_sample):
    """Plot SHAP feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Bar plot
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig("reports/Script_Visuals/shap_importance_bar_reg.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Beeswarm plot
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig("reports/Script_Visuals/shap_importance_beeswarm_reg.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==============================
# 3. BACKTESTING (PnL Simulation)
# ==============================
def backtest_regression(df_test, model, features, slippage=0.0005):
    """Backtest continuous PnL predictions."""
    df = df_test.copy()
    df['predicted_pnl'] = model.predict(df[features])
    df['predicted_pnl'] *= (1 - slippage)  # Apply slippage

    equity = 100000
    equity_curve = []
    returns = []

    for pnl in df['predicted_pnl']:
        equity += pnl
        returns.append(pnl / 100000)
        equity_curve.append(equity)

    df['equity'] = equity_curve
    df['returns'] = returns

    sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252)
    max_dd = (pd.Series(equity_curve) / pd.Series(equity_curve).cummax() - 1).min()

    metrics = {
        'final_equity': equity,
        'sharpe': sharpe,
        'max_drawdown_pct': max_dd * 100
    }
    return df, metrics

# ==============================
# 4. MODEL DEPLOYMENT
# ==============================
def save_model(model, path):
    joblib.dump(model, path)
    print(f"Regression model saved to {path}")

def load_model(path):
    return joblib.load(path)


