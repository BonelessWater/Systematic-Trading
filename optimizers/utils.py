import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_tracking_error_metrics_scalar(daily_data):
    """
    Calculate scalar tracking error metrics for the entire trading timeline.

    Parameters:
    - daily_data (pd.DataFrame): A DataFrame with columns ['date', 'ideal', 'realized'].

    Returns:
    - metrics (dict): A dictionary containing scalar values for each metric.
    """
    if daily_data.empty or 'ideal' not in daily_data.columns or 'realized' not in daily_data.columns:
        raise ValueError("Input DataFrame is empty or missing required columns 'ideal' and 'realized'.")

    # Drop rows with missing values in 'ideal' or 'realized'
    daily_data = daily_data.dropna(subset=['ideal', 'realized'])

    # Initialize variables to accumulate values
    total_mate = 0  # Mean Absolute Tracking Error
    total_rmste = 0  # Root Mean Square Tracking Error
    max_te = 0  # Maximum Tracking Error
    total_plte = 0  # Portfolio-Level Tracking Error
    total_weights = 0  # For normalization

    for date, group in daily_data.groupby('date'):
        ideal_positions = np.array(group['ideal'], dtype=np.float64)
        realized_positions = np.array(group['realized'], dtype=np.float64)
        tracking_errors = realized_positions - ideal_positions

        # Skip groups with no valid data
        if len(ideal_positions) == 0:
            continue

        # Metrics for the group
        mate = np.mean(np.abs(tracking_errors))
        rmste = np.sqrt(np.mean(tracking_errors**2))
        max_te = max(max_te, np.max(np.abs(tracking_errors)))
        plte = np.sqrt(np.sum(tracking_errors**2))

        # Accumulate values
        total_mate += mate * len(group)
        total_rmste += rmste * len(group)
        total_plte += plte
        total_weights += len(group)

    # Avoid division by zero
    if total_weights == 0:
        raise ValueError("No valid data to calculate metrics.")

    # Compute scalars
    return {
        'MATE': total_mate / total_weights,  # Weighted Mean Absolute Tracking Error
        'RMSTE': total_rmste / total_weights,  # Weighted Root Mean Square Tracking Error
        'MaxTE': max_te,  # Overall Maximum Tracking Error
        'PLTE': total_plte  # Total Portfolio-Level Tracking Error
    }
