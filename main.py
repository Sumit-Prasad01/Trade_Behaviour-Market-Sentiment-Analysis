
"""
Main pipeline to run:
1. Feature engineering
2. Train/test split
3. Classification: baseline & advanced
4. Regression: baseline & advanced
5. Evaluation & backtesting
"""

import pandas as pd
from scripts.feature_engineering_01 import make_features
from scripts.utils_split_eval_02 import time_train_test_split, evaluate_classification, evaluate_regression

# Classification functions
from scripts.model_classification_03 import (
    train_classifiers,
    handle_class_imbalance,
    tune_xgb_classifier,
    plot_shap_importance,
    calibrate_model,
    backtest,
    save_model
)

# Regression functions
from scripts.model_regression_04 import (
    train_regressors,
    tune_xgb_regressor,
    plot_shap_importance as plot_shap_reg,
    backtest_regression,
    save_model as save_model_reg
)

# ==============================
# 1. LOAD & FEATURE ENGINEERING
# ==============================
df = pd.read_csv("data/Merged Data/merged_trades_sentiment.csv", parse_dates=['date'])
df, features = make_features(df)

# ==============================
# 2. TRAIN/TEST SPLIT (TIME-AWARE)
# ==============================
train, test = time_train_test_split(df, test_size_days=90)

# ------------------------------
# CLASSIFICATION TARGET: 'win'
# ------------------------------
X_train, y_train = train[features], train['win']
X_test, y_test = test[features], test['win']

print("\n=== CLASSIFICATION MODELS ===")
# Baseline models
models_clf = train_classifiers(X_train, y_train)
print("\n--- Baseline Classification Results ---")
for name, m in models_clf.items():
    y_pred = m.predict(X_test)
    try:
        y_proba = m.predict_proba(X_test)[:, 1]
    except:
        y_proba = None
    print(name, evaluate_classification(y_test, y_pred, y_proba))

# Advanced tuned model
print("\n--- Advanced Tuned XGBoost ---")
X_train_bal, y_train_bal = handle_class_imbalance(X_train, y_train)
best_model_clf = tune_xgb_classifier(X_train_bal, y_train_bal)
plot_shap_importance(best_model_clf, X_train.sample(200))
cal_model_clf = calibrate_model(best_model_clf, X_train, y_train, X_test, y_test)
bt_df_clf, bt_metrics_clf = backtest(test, cal_model_clf, features)
print("Backtest Metrics:", bt_metrics_clf)
save_model(cal_model_clf, "models/xgb_win.pkl")

# ------------------------------
# REGRESSION TARGET: 'Closed PnL'
# ------------------------------
print("\n=== REGRESSION MODELS ===")
X_train_r, y_train_r = train[features], train['closed_pnl']
X_test_r, y_test_r = test[features], test['closed_pnl']

# Baseline models
models_reg = train_regressors(X_train_r, y_train_r)
print("\n--- Baseline Regression Results ---")
for name, m in models_reg.items():
    y_pred_r = m.predict(X_test_r)
    print(name, evaluate_regression(y_test_r, y_pred_r))

# Advanced tuned model
print("\n--- Advanced Tuned XGBoost Regressor ---")
best_model_reg = tune_xgb_regressor(X_train_r, y_train_r)
plot_shap_reg(best_model_reg, X_train_r.sample(200))
bt_df_reg, bt_metrics_reg = backtest_regression(test, best_model_reg, features)
print("Backtest Metrics:", bt_metrics_reg)
save_model_reg(best_model_reg, "models/xgb_reg_pnl.pkl")
