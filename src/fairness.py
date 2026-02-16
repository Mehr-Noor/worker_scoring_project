# فایل: src/fairness.py

import pandas as pd
import numpy as np

def demographic_parity(df: pd.DataFrame, group_col: str, decision_col: str = "decision") -> dict:
    """
    Calculate Demographic Parity (rate of 'Pass') per group
    """
    rates = {}
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        rates[group] = (group_df[decision_col] == "Pass").mean()
    return rates

def disparate_impact(df: pd.DataFrame, group_col: str, reference_group: str, decision_col: str = "decision") -> dict:
    """
    Disparate Impact = rate(group) / rate(reference_group)
    """
    rates = demographic_parity(df, group_col, decision_col)
    ref_rate = rates.get(reference_group)

    if ref_rate is None or ref_rate == 0:
        raise ValueError(f"Reference group '{reference_group}' not found or has zero Pass rate.")

    impact = {}
    for group, rate in rates.items():
        impact[group] = rate / ref_rate

    return impact
