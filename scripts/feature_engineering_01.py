import pandas as pd
import numpy as np

def make_features(df, date_col='date'):
    """
    Input: merged trades+sentiment df with columns:
           ['date','Closed PnL','Size USD','value' (sentiment score),'classification','leverage','Side','Account', ...]
    Output: df with new features (rolling sentiment, vol, streaks, encodings)
    """

    df = df.copy()
    # ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Basic target
    df['win'] = (df['closed_pnl'] > 0).astype(int)

    # Time-based features
    df['dayofweek'] = df[date_col].dt.weekday
    df['hour'] = (df[date_col].dt.hour if 'timestamp_utc' in df.columns else 0)
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

    # Size/price features
    df['size_usd'] = pd.to_numeric(df.get('Size USD', df.get('size_usd', np.nan)), errors='coerce')
    df['abs_size_usd'] = df['size_usd'].abs()

    # Sentiment features (daily series aggregated)
    # If sentiment is daily and multiple trades per day, compute per-day sentiment and then map:
    daily_sent = df.groupby(df[date_col].dt.date).agg(
        sentiment_val=('value', 'mean')
    )
    daily_sent.index = pd.to_datetime(daily_sent.index)
    df['sentiment_val'] = df[date_col].dt.date.map(lambda d: daily_sent.loc[pd.to_datetime(d),'sentiment_val'])

    # Rolling sentiment stats (use past N days, shift to avoid leakage)
    # Create a timeseries of daily sentiment to compute rolling stats, then map back to trades
    s = daily_sent['sentiment_val'].asfreq('D').interpolate()
    s_7 = s.rolling(7, min_periods=1).mean().shift(1)   # 7-day rolling mean using past window (shift 1)
    s_3 = s.rolling(3, min_periods=1).mean().shift(1)
    s_vol_7 = s.diff().abs().rolling(7, min_periods=1).std().shift(1)

    # Map rolling features back to trades
    df['sent_7d'] = df[date_col].dt.date.map(lambda d: s_7.loc[pd.to_datetime(d)] if pd.to_datetime(d) in s_7.index else np.nan)
    df['sent_3d'] = df[date_col].dt.date.map(lambda d: s_3.loc[pd.to_datetime(d)] if pd.to_datetime(d) in s_3.index else np.nan)
    df['sent_vol_7d'] = df[date_col].dt.date.map(lambda d: s_vol_7.loc[pd.to_datetime(d)] if pd.to_datetime(d) in s_vol_7.index else np.nan)

    # Recent PnL streaks per account (rolling win rate over last N trades)
    df = df.sort_values([ 'account', date_col ])
    df['acc_win'] = df.groupby('account')['win'].transform(lambda x: x.rolling(window=20, min_periods=1).mean().shift(1))
    df['acc_avg_pnl_20'] = df.groupby('account')['closed_pnl'].transform(lambda x: x.rolling(window=20, min_periods=1).mean().shift(1))

    # Side encoding
    df['side_buy'] = (df['side'].str.upper() == 'BUY').astype(int)

    # One-hot encode sentiment classification (use prefix sent_)
    df['classification'] = df['classification'].fillna('Unknown')
    sent_dummies = pd.get_dummies(df['classification'], prefix='sent')
    df = pd.concat([df, sent_dummies], axis=1)

    # Fill NA in engineered features with reasonable values
    fill_cols = ['sent_7d','sent_3d','sent_vol_7d','acc_win','acc_avg_pnl_20','abs_size_usd']
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Optionally create ratio/interaction features
    df['pnl_per_notional'] = df['closed_pnl'] / (df['abs_size_usd'].replace({0:np.nan}))
    df['pnl_per_notional'] = df['pnl_per_notional'].fillna(0)

    # final features list (return df and feature names)
    feature_cols = [
        'abs_size_usd','leverage','side_buy','sent_7d','sent_3d','sent_vol_7d',
        'acc_win','acc_avg_pnl_20','pnl_per_notional','dayofweek','is_weekend'
    ]
    # include one-hot columns
    feature_cols += list(sent_dummies.columns)
    # keep only existing
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df, feature_cols

