import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ==============================
# 0. BASELINE MODELS
# ==============================

def train_classifiers(X_train, y_train):
    """Trains baseline Logistic Regression & default XGBoost."""
    # Logistic regression baseline
    pipe_lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipe_lr.fit(X_train, y_train)

    # XGBoost baseline
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_train, y_train)

    return {'logistic': pipe_lr, 'xgboost': xgb}

# ==============================
# 1. CLASS IMBALANCE HANDLING
# ==============================
def handle_class_imbalance(X_train, y_train):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

# ==============================
# 2. HYPERPARAMETER TUNING
# ==============================
def tune_xgb_classifier(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=5)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    grid = GridSearchCV(
        xgb,
        param_grid,
        scoring='roc_auc',
        cv=tscv,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print("Best Params:", grid.best_params_)
    return grid.best_estimator_

# ==============================
# 3. FEATURE IMPORTANCE (SHAP)
# ==============================
def plot_shap_importance(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig("reports/Script_Visuals/shap_importance_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig("reports/Script_Visuals/shap_importance_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==============================
# 4. PROBABILITY CALIBRATION
# ==============================
def calibrate_model(model, X_train, y_train, X_test, y_test):
    tscv = TimeSeriesSplit(n_splits=3)
    cal_model = CalibratedClassifierCV(model, method='isotonic', cv=tscv)
    cal_model.fit(X_train, y_train)

    prob_true, prob_pred = calibration_curve(y_test, cal_model.predict_proba(X_test)[:, 1], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title("Calibration Curve")
    plt.savefig("reports/Script_Visuals/calibration_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    return cal_model

# ==============================
# 5. ROBUST BACKTESTING
# ==============================
def backtest(df_test, model, features, prob_threshold=0.6, slippage=0.0005):
    df = df_test.copy()
    probs = model.predict_proba(df[features])[:, 1]
    df['signal'] = (probs >= prob_threshold).astype(int)

    equity = 100000
    equity_curve = []
    returns = []

    for _, row in df.iterrows():
        if row['signal'] == 1:
            pnl = row['closed_pnl'] * (1 - slippage)
            equity += pnl
            returns.append(pnl / 100000)
        else:
            returns.append(0)
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
# 6. MODEL DEPLOYMENT
# ==============================
def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    return joblib.load(path)



