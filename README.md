# 📊 Trade Behaviour & Market Sentiment Analysis

This project explores the relationship between **Bitcoin market sentiment** and **trader performance** using historical trade data from Hyperliquid and a Fear & Greed sentiment dataset.  
We perform **data cleaning, EDA, predictive modeling, and strategy backtesting**, and finally deploy **interactive dashboards** for insights and predictions.

---

## 📁 Project Structure

```
trader_sentiment_project/
│
├── dashboard/
|   └──streamlit_app.py # Eda dashboard using streamlit
├── app
|    ├── app.py             # Prediction using models 
|    └── train_app.py       # Train the models
├── data/
│   ├── raw/                # Original datasets
|   ├── Merged Data         # merged datasets
|   ├── Sentiment Data      # Sentiment EDA Data
|   ├── CLuster             #  Trade & Sentiment Cluster Data
│   └── preprocessed/       # Cleaned datasets
│
├── notebooks/
│   ├── 01_Historical_Data_Cleaning.ipynb
│   ├── 02_Sentiment_Data_Cleaning.ipynb
│   ├── 03_Historical_Data_EDA.ipynb
│   ├── 04_Sentiment_EDA.ipynb
│   ├── 05_Merge_trade_sentiment_EDA.ipynb
│   ├── 06_Combined_trade&sentiment_EDA.ipynb
│   └── 07_Advanced_Combined_EDA.ipynb
│   
│
├── reports/                # Saved charts from EDA & ML
│   ├── Combined_EDA 
|   ├── Scripts_Visuals        
│   └── Seperate EDA
│
├── scripts/
|   ├── feature_engineering_01.py
|   ├── utils_split_eval_02.py
|   ├── model_classification.py
|   ├── model_regression_04.py
│   └── backtest_05.py               
│
├── models/
|   ├── xgb_reg_pnl.pkl
│   └── xgb_win.pkl         
│
├── requirements.txt         # Python dependencies
├── Final_Report.md          # Final Project Report 
├── Model_Summary.md         # Model Summary
├── README.md                # Project description & instructions
├── main.py                  # main python script
└── .gitignore               # Ignore venv, data/raw, etc.

```
## 📌 Features

### 1. **Data Preparation**
- Separate cleaning pipelines for **trades** and **sentiment** datasets.
- Merging datasets on **date** for combined analysis.
- Duplicate removal, timestamp formatting, missing value handling.

### 2. **EDA (Exploratory Data Analysis)**
- **Trade EDA**: trade volume patterns, leverage trends, PnL distribution, account clustering.
- **Sentiment EDA**: sentiment distribution, time series decomposition, seasonal patterns.
- **Combined EDA**: correlation between sentiment categories & trader PnL, volatility analysis.
- **Advanced**: statistical hypothesis testing, outlier detection, clustering by behaviour.

### 3. **Predictive Modeling**
- **Classification**: predict win/loss probability.
- **Regression**: predict expected PnL.
- **Techniques**: Logistic Regression, XGBoost, LightGBM.
- **Enhancements**:
  - Hyperparameter tuning with `TimeSeriesSplit`.
  - Class imbalance handling (SMOTE, class weights).
  - SHAP feature importance analysis.
  - Probability calibration for classification models.

### 4. **Backtesting & Strategy Simulation**
- Rule-based simulations (e.g., trade only in "Fear" phases).
- Metrics: Sharpe Ratio, Max Drawdown, Win Rate.
- Adjustable parameters: leverage, slippage, liquidity limits.

### 5. **Dashboards**
- **EDA Dashboard** (`app_analysis.py`):
  - Interactive trade/sentiment plots.
  - Combined heatmaps & correlations.
  - Backtest simulations with user-set parameters.
- **Prediction Dashboard** (`app_prediction.py`):
  - Upload trained model & dataset for predictions.
  - Classification: win probability & distribution plots.
  - Regression: PnL predictions & distribution.

---
## 📈 Results & Insights
- Strong correlation between extreme sentiment states and trader performance.

- Certain accounts perform significantly better in "Fear" markets.

- Strategy simulation based on sentiment phases showed improved Sharpe ratios over baseline.

🛠 Technologies Used
- Python: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, shap

- Visualization: matplotlib, seaborn, plotly

- Dashboards: Streamlit

- Backtesting: custom Python logic

- Version Control: Git

## - 📅 Project Flow
- Data Collection

- Cleaning & Preprocessing

- EDA & Insight Generation

- Feature Engineering

- Modeling & Evaluation

- Backtesting

- Dashboard Deployment

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/trade-sentiment-analysis.git
cd trade-sentiment-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Streamlit App
- App Link - https://sumit-prasad01-trade-behaviour-ma-dashboardstreamlit-app-o97aco.streamlit.app/

####*** This App does not contain full features of project (analysis).This is just for demo.***####