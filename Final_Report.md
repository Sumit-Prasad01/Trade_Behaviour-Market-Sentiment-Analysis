# Final Project Report — Trader Performance vs Market Sentiment

## 1. Project Overview
This project explores the relationship between **trader performance** and **Bitcoin market sentiment** to uncover patterns and actionable insights. The analysis includes:
- Data cleaning & preprocessing for two datasets: historical trades and market sentiment.
- Separate EDA for trades and sentiment.
- Combined EDA to identify relationships between sentiment and PnL.
- Advanced statistical analysis.
- Predictive modeling & backtesting for strategy simulation.

---

## 2. Datasets

### **2.1 Historical Trader Data**
- **Columns**: Account, Coin, Execution Price, Size USD, Side, Timestamp, Leverage, Closed PnL, Fee, etc.
- **Source**: Hyperliquid exchange.
- **Frequency**: Trade-level.

### **2.2 Bitcoin Market Sentiment**
- **Columns**: Timestamp, Value (0–100), Classification (Fear, Greed, etc.), Date.
- **Source**: Fear & Greed Index.
- **Frequency**: Daily.

---

## 3. Data Cleaning & Preprocessing

### **3.1 Historical Trades**
- Converted timestamps to UTC & IST.
- Removed exact duplicate rows.
- Handled missing leverage, fees, and PnL values.
- Derived **abs_size_usd**, normalized PnL per notional value.
- Added weekday/weekend indicators.

### **3.2 Sentiment Data**
- Converted Unix timestamps to dates.
- Handled missing sentiment values with linear interpolation.
- Standardized classification labels.
- Created rolling averages and sentiment volatility metrics.

---

## 4. Exploratory Data Analysis (EDA)

### **4.1 Trades**
- **PnL Distribution**: Skewed with heavy tails; majority of trades have small gains/losses.
- **Volume Patterns**: Higher trade volume in weekdays vs weekends.
- **Leverage Usage**: High leverage linked to larger PnL variance.
- **Account Clustering**: KMeans reveals trader segments by size, leverage, and win rate.

### **4.2 Sentiment**
- **Trend**: Clear long-term cycles between fear and greed.
- **Seasonal Decomposition**: Yearly periodicity detected.
- **Classification Analysis**: Extreme Fear often correlates with high volatility.
- **Markov Chain**: Sentiment states are persistent, especially in Fear phases.

### **4.3 Combined Analysis**
- **PnL vs Sentiment**: Certain sentiment phases (Extreme Fear) show higher average PnL for BUY trades.
- **Leverage vs Sentiment**: Leverage spikes in Extreme Greed periods.
- **Lag Analysis**: 1–2 day lag in sentiment change sometimes precedes performance shift.
- **Cross-Coin Effect**: BTC sentiment impacts altcoin trades.

---

## 5. Advanced Analysis

### **5.1 Hypothesis Testing**
- ANOVA & t-tests confirm significant differences in mean PnL between sentiment categories.

### **5.2 Outlier Detection**
- Large positive/negative PnL trades identified; many occur during sentiment extremes.

### **5.3 Cross-Coin Sentiment Impact**
- Heatmaps reveal sentiment phases influencing performance across multiple coins.

---

## 6. Predictive Modeling

### **6.1 Feature Engineering**
- Rolling sentiment averages (3-day, 7-day).
- Sentiment volatility.
- One-hot encoding of sentiment categories.
- Account-level rolling win rate & avg PnL.
- Trade volume, leverage, weekday/weekend indicators.

### **6.2 Model Selection**
- **Classification**: Logistic Regression, XGBoost, Random Forest.
- **Regression**: XGBoostRegressor, LightGBM.

---

## 7. Six Practical Steps Implemented

1. **Hyperparameter Tuning**: GridSearch with TimeSeriesSplit.
2. **Class Imbalance Handling**: Class weights & SMOTE for training set.
3. **Feature Importance**: SHAP analysis for model explainability.
4. **Probability Calibration**: Isotonic calibration + reliability curves.
5. **Robust Backtesting**: Included slippage & liquidity constraints.
6. **Model Deployment**: Saved models with joblib for live use.

---

## 8. Strategy Simulation & Backtesting

### **Rules Tested**
- Go long only in Fear & Neutral phases.
- Reduce leverage during Extreme Greed.

### **Metrics**
- **Sharpe Ratio**: Improved under sentiment-aware strategy.
- **Max Drawdown**: Reduced compared to baseline.
- **Win Rate**: Increased in targeted phases.

---

## 9. Key Insights

- Sentiment extremes (both fear & greed) often precede high-volatility trade opportunities.
- Leverage tends to spike during greed phases, increasing risk.
- BUY trades in Fear phases have historically higher PnL averages.
- Lag effects suggest sentiment shifts can be predictive signals.

---

## 10. Recommendations

- Incorporate sentiment features into trading strategy.
- Limit leverage during Extreme Greed periods to reduce drawdowns.
- Explore automated trading signals based on lagged sentiment changes.
- Maintain monitoring via regular EDA updates to adjust strategy over time.

---

## 11. Next Steps

- Expand dataset to include more assets for broader sentiment correlation.
- Experiment with deep learning models (e.g., LSTMs) for sentiment-based forecasting.
- Build a real-time dashboard for live monitoring.

---
**End of Report**
