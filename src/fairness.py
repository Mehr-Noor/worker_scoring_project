# فایل: src/fairness.py

import pandas as pd
from sklearn.metrics import confusion_matrix

def demographic_parity(df, group_col, decision_col="decision"):
    rates = {}
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        pass_rate = (group_df[decision_col] == "Pass").mean()
        rates[group] = round(pass_rate, 3)
    return rates


def equal_opportunity(df, group_col, y_true_col="decision", y_pred_col="decision"):
    results = {}
    for group in df[group_col].unique():
        g = df[df[group_col] == group]
        y_true = (g[y_true_col] == "Pass").astype(int)
        y_pred = (g[y_pred_col] == "Pass").astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        results[group] = round(tpr, 3)

    return results


def false_negative_rate(df, group_col, y_true_col="decision", y_pred_col="decision"):
    results = {}
    for group in df[group_col].unique():
        g = df[df[group_col] == group]
        y_true = (g[y_true_col] == "Pass").astype(int)
        y_pred = (g[y_pred_col] == "Pass").astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        results[group] = round(fnr, 3)

    return results
