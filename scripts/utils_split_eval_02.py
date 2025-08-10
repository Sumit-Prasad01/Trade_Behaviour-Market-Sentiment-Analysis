from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd


def time_train_test_split(df, date_col='date', test_size_days=90):
    df = df.sort_values(date_col)
    max_date = df[date_col].max()
    test_start = max_date - pd.Timedelta(days=test_size_days)
    train = df[df[date_col] < test_start].copy()
    test = df[df[date_col] >= test_start].copy()
    return train, test

def evaluate_classification(y_true, y_pred, y_proba=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_proba is not None:
        try:
            results['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            results['roc_auc'] = None
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    return results

def evaluate_regression(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        'rmse': mean_squared_error(y_true, y_pred, squared=False),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
