import numpy as np
import pandas as pd

def backtest_signals(df_test, model, features, stake_fraction=0.01, equity_init=100000, prob_threshold=0.6, fee_pct=0.0005):
    """
    df_test: test dataframe (sorted by date) with trade rows and columns including 'abs_size_usd', 'Closed PnL' etc.
    model: trained classifier with .predict_proba
    features: list of feature column names
    stake_fraction: fraction of current equity to risk per signal
    prob_threshold: threshold to go long (only long example)
    fee_pct: transaction fee proportion applied on notional
    """
    df = df_test.copy().reset_index(drop=True)
    X = df[features]
    probs = model.predict_proba(X)[:,1]
    df['pred_prob'] = probs
    df['signal'] = (df['pred_prob'] >= prob_threshold).astype(int)

    equity = equity_init
    equity_curve = []
    returns = []
    position_sizes = []

    for idx, row in df.iterrows():
        signal = row['signal']
        notional = row['abs_size_usd'] if not np.isnan(row['abs_size_usd']) else 0
        # position sizing: take min of stake_fraction*equity and trade notional
        pos_size = min(equity * stake_fraction, notional)
        position_sizes.append(pos_size)

        if signal == 1 and pos_size > 0:
            # scale actual trade pnl proportionally if model uses whole trade pnl
            # e.g., assume Closed PnL scales linearly with position size:
            actual_pnl = row['Closed PnL'] * (pos_size / (notional if notional>0 else 1))
            fee = pos_size * fee_pct
            pnl_net = actual_pnl - fee
            equity += pnl_net
            returns.append(pnl_net / equity_init)
        else:
            returns.append(0)
        equity_curve.append(equity)

    df['equity'] = equity_curve
    df['return'] = returns

    # Metrics
    net_return = (equity_curve[-1] - equity_init) / equity_init
    daily_returns = pd.Series(returns)
    if len(daily_returns) > 1:
        sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)
    else:
        sharpe = np.nan

    # Max drawdown
    eq = pd.Series(equity_curve)
    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max
    max_dd = drawdown.min()

    win_rate = (pd.Series(returns) > 0).sum() / (pd.Series(returns) != 0).sum() if (pd.Series(returns) != 0).sum()>0 else 0

    summary = {
        'net_return_pct': net_return * 100,
        'sharpe': sharpe,
        'max_drawdown_pct': max_dd * 100,
        'final_equity': equity_curve[-1],
        'win_rate': win_rate
    }

    return df, summary
