# Modeling Summary – Trade Behaviour & Market Sentiment Analysis
## 1. Objective
- The modeling stage aimed to predict trader performance (win/loss classification) and forecast PnL values (regression) based on market sentiment and trade features.
- This enables building data-driven trading strategies.

## 2. Data Preparation
- Input: Cleaned & processed merged_trades_sentiment.csv from data/processed/

- Features: Engineered from trade + sentiment data (feature_engineering_01.py)

- Targets:

    - Classification: win (1 = profitable trade, 0 = loss)

    - Regression: Closed PnL (continuous)

## 3. Model Types
### Classification Models
#### 1.Baseline Models

- Logistic Regression (with imputation & scaling)

- XGBoost Classifier (default parameters)

####  2.Advanced Tuned Model

- XGBoost Classifier with hyperparameter tuning:

    - Parameters tuned: n_estimators, max_depth, learning_rate, subsample

    - Time-series aware cross-validation (TimeSeriesSplit)

- Class imbalance handled via SMOTE oversampling

- Probability calibration using isotonic regression

- Feature importance via SHAP

- Strategy backtesting with:

    - Position sizing

    - Slippage

    - Liquidity constraints

### Regression Models
#### 1.Baseline Models

- Linear Regression (with imputation & scaling)

- Random Forest Regressor

- XGBoost Regressor (default parameters)

#### 2.Advanced Tuned Model

- XGBoost Regressor with hyperparameter tuning:

    - Same parameter grid as classification model

- Feature importance via SHAP

- PnL simulation backtest with slippage

## 4. Evaluation Metrics
### Classification
- Accuracy, Precision, Recall, F1-score

- ROC-AUC for probability quality

- Backtest metrics:

    - Sharpe Ratio: risk-adjusted return measure

    - Max Drawdown: worst portfolio decline from peak

### Regression
- RMSE, MAE

- Backtest metrics:

    - Final Equity (starting from 100k)

    - Sharpe Ratio

    - Max Drawdown %

## 5. Key Insights
- Logistic Regression served as a transparent baseline, but lacked predictive power compared to XGBoost.

- XGBoost (tuned) consistently outperformed baselines in both accuracy and financial metrics.

- Sentiment volatility and recent PnL streaks ranked high in SHAP feature importance.

- Calibrated probabilities improved backtest performance by reducing false high-confidence signals.

- Regression models successfully forecasted PnL magnitude, aiding risk-adjusted position sizing.

## 6. Model Deployment
- Best models saved as:

    - models/xgb_win.pkl → classification

    - models/xgb_reg_pnl.pkl → regression