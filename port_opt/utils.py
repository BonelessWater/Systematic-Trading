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
    # Initialize variables to accumulate values
    total_mate = 0  # Mean Absolute Tracking Error
    total_rmste = 0  # Root Mean Square Tracking Error
    max_te = 0  # Maximum Tracking Error
    total_plte = 0  # Portfolio-Level Tracking Error
    total_weights = 0  # For normalization

    for date, group in daily_data.groupby('date'):
        ideal_positions = np.array(group['ideal'])
        realized_positions = np.array(group['realized'])
        tracking_errors = realized_positions - ideal_positions

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

    # Compute scalars
    return {
        'MATE': total_mate / total_weights,  # Weighted Mean Absolute Tracking Error
        'RMSTE': total_rmste / total_weights,  # Weighted Root Mean Square Tracking Error
        'MaxTE': max_te,  # Overall Maximum Tracking Error
        'PLTE': total_plte  # Total Portfolio-Level Tracking Error
    }

def calculate_optimizer_quality(metrics, weights=None):
    """
    Calculate a single scalar value to assess optimizer quality.

    Parameters:
    - metrics (pd.DataFrame): A DataFrame with tracking error metrics (e.g., MATE, RMSTE, MaxTE, PLTE).
    - weights (dict): Weights for each metric in the form {metric_name: weight}. Default is equal weighting.

    Returns:
    - quality_score (float): A single scalar value representing the optimizer's quality.
    """

    # Compute weighted aggregate score
    quality_score = 0.0
    for key, value in metrics.items():
        quality_score += value

    return quality_score
