import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="Trade & Sentiment Analysis Dashboard", layout="wide")
sns.set_style("whitegrid")

# ==========================
# DATA LOADER
# ==========================
@st.cache_data
def load_default_data():
    trades = pd.read_csv("../data/preprocessed/historical_data/cleaned_historical_data.csv", parse_dates=['date'])
    sentiment = pd.read_csv("../data/preprocessed/sentiment_data/sentiment_clean.csv", parse_dates=['date'])
    merged = pd.read_csv("../data/Merged Data/merged_trades_sentiment.csv", parse_dates=['date'])
    return trades, sentiment, merged

def load_uploaded_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['date'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ==========================
# BACKTEST FUNCTION
# ==========================
def backtest_strategy(df, prob_threshold=0.6, slippage=0.001, position_size_pct=0.05, start_balance=100000):
    df = df.copy()
    if 'pred_prob' not in df.columns or 'Closed PnL' not in df.columns:
        st.warning("Dataset must contain 'pred_prob' and 'Closed PnL' columns for backtesting.")
        return None, None

    equity = start_balance
    equity_curve = []

    for _, row in df.iterrows():
        if row['pred_prob'] > prob_threshold:
            trade_size = equity * position_size_pct
            pnl = row['Closed PnL']
            pnl -= abs(trade_size) * slippage
            equity += pnl
        equity_curve.append(equity)

    sharpe = (pd.Series(equity_curve).pct_change().mean() /
              pd.Series(equity_curve).pct_change().std()) * (252 ** 0.5)
    max_dd = ((max(equity_curve) - min(equity_curve)) / max(equity_curve)) * 100

    results = {
        "Final Equity": round(equity, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Max Drawdown %": round(max_dd, 2)
    }
    return results, equity_curve

# ==========================
# LOAD DATA
# ==========================
trades_df, sentiment_df, merged_df = load_default_data()

st.sidebar.header("Upload Data (optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
if uploaded_file is not None:
    user_df = load_uploaded_data(uploaded_file)
    if user_df is not None:
        st.session_state["uploaded_df"] = user_df

# ==========================
# MAIN TABS
# ==========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Trade EDA", "📈 Sentiment EDA", "📉 Combined EDA", "🎯 Backtest Simulation"])

# ==========================
# TAB 1 – TRADE EDA
# ==========================
with tab1:
    st.title("📊 Trade EDA")
    df = trades_df if "uploaded_df" not in st.session_state else st.session_state["uploaded_df"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", len(df))
    col2.metric("Unique Accounts", df['Account'].nunique())
    col3.metric("Avg Closed PnL", f"{df['Closed PnL'].mean():.2f}")

    fig, ax = plt.subplots()
    sns.histplot(df['Closed PnL'], bins=50, kde=True, ax=ax)
    ax.set_title("Closed PnL Distribution")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    df.groupby('date')['Closed PnL'].mean().plot(ax=ax)
    ax.set_title("Daily Average PnL")
    st.pyplot(fig)

# ==========================
# TAB 2 – SENTIMENT EDA
# ==========================
with tab2:
    st.title("📈 Sentiment EDA")
    df = sentiment_df if "uploaded_df" not in st.session_state else st.session_state["uploaded_df"]

    col1, col2 = st.columns(2)
    col1.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")
    col2.metric("Avg Sentiment Score", f"{df['value'].mean():.2f}")

    fig, ax = plt.subplots()
    sns.lineplot(x='date', y='value', data=df, ax=ax)
    ax.set_title("Sentiment Score Over Time")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    df['classification'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Sentiment Classification Counts")
    st.pyplot(fig)

# ==========================
# TAB 3 – COMBINED EDA
# ==========================
with tab3:
    st.title("📉 Combined Insights")
    df = merged_df if "uploaded_df" not in st.session_state else st.session_state["uploaded_df"]

    fig, ax = plt.subplots()
    df.groupby('classification')['Closed PnL'].mean().plot(kind='bar', ax=ax)
    ax.set_title("Average PnL by Sentiment Classification")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.scatterplot(x='value', y='Closed PnL', data=df, ax=ax)
    ax.set_title("Sentiment Score vs Closed PnL")
    st.pyplot(fig)

# ==========================
# TAB 4 – BACKTEST
# ==========================
with tab4:
    st.title("🎯 Backtest Simulation")
    df = merged_df if "uploaded_df" not in st.session_state else st.session_state["uploaded_df"]

    prob_threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.6, 0.01)
    slippage = st.number_input("Slippage %", 0.0, 5.0, 0.1) / 100
    position_size_pct = st.number_input("Position Size %", 0.0, 1.0, 0.05)
    start_balance = st.number_input("Starting Balance", 1000, 1000000, 100000)

    if st.button("Run Backtest"):
        results, equity_curve = backtest_strategy(
            df,
            prob_threshold=prob_threshold,
            slippage=slippage,
            position_size_pct=position_size_pct,
            start_balance=start_balance
        )
        if results:
            st.write("### Backtest Results")
            st.write(results)

            fig, ax = plt.subplots()
            ax.plot(equity_curve)
            ax.set_title("Equity Curve")
            st.pyplot(fig)
