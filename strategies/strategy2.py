from strategies.base_strategy import BaseStrategy
from risk_management.risk_obj import Risk
from scipy.stats import norm
import pandas as pd
import numpy as np
import os
from typing import Callable, Any, List, Dict
from risk_management.constraints import (
    positive_capital_constraint, 
    max_leverage_constraint,
    minimum_volatility_constraint, 
    garch_constraint, 
    portfolio_risk_constraint, 
    jump_risk_constraint, 
    concentration_constraint,
    liquidity_constraint
)
from risk_management.metrics import (
    portfolio_volatility_metric,
    sharpe_ratio_metric,
    sortino_ratio_metric,
    value_at_risk_metric,
    beta_metric,
    conditional_value_at_risk_metric,
    greek_values_metric,
    upside_potential_ratio_metric
)

class Strategy2Risk(Risk):
    """
    Custom Risk implementation for Strategy2.
    """
    def __init__(self, data, capital, covariance_matrices, lookback_period=20):
        # Initialize the base class with data and capital
        super().__init__(data, capital)
        self.lookback_period = lookback_period
        self.covariance_matrices = covariance_matrices  # Add covariance matrices

    def base_constraints(self, data: dict, positions) -> bool:
        """
        Define base constraints for Strategy1.
        """
        # Constraint: Ensure remaining capital is positive
        remaining_capital = self.capital - (positions * data["close"]).sum()
        return remaining_capital > 0

    def base_metrics(self, data: dict) -> dict:
        """
        Define base metrics for Strategy1.
        """
        # Metric: Portfolio standard deviation (volatility)
        # Evaluate and return metrics
            # Define a dynamic get_returns function
        # get_returns = lambda d: d["Daily_Return"]

        # Evaluate and return metrics
        volatility_metric = portfolio_volatility_metric()  # Get the closure

    # Evaluate the metric using the closure
        metrics = volatility_metric(data)  # Call the closure with `data`
        return metrics

        # metrics = portfolio_volatility_metric(data)
        # return metrics

class Strategy2(BaseStrategy):
    def __init__(self, data, risk_target, capital, num_stocks, lookback_period=20):
        """
        Initializes Strategy2 with a custom Risk object.
        """
        super().__init__(data, risk_target, capital, num_stocks)
        self.positions = None  # Positions will be calculated later
        self.lookback_period = lookback_period  # Set lookback_period as an attribute
        self.risk = Strategy2Risk(data, capital, self.covariance_matrices)  # Pass covariance matrices
                # Add metrics

        self.risk.add_metric(portfolio_volatility_metric())
        self.risk.add_metric(sharpe_ratio_metric(risk_free_rate=0.02))
        self.risk.add_metric(sortino_ratio_metric(risk_free_rate=0.01))
        self.risk.add_metric(value_at_risk_metric(confidence_level=0.95))
        self.risk.add_metric(conditional_value_at_risk_metric(confidence_level=0.95))
        self.risk.add_metric(beta_metric())
        self.risk.add_metric(
            greek_values_metric(
                underlying_price=100,
                strike_price=100,
                time_to_expiry=1,  # 1 year
                risk_free_rate=0.02,
                volatility=0.2
            )
        )

        # Add constraints
        # self.risk.add_constraint(positive_capital_constraint())
        self.risk.add_constraint(max_leverage_constraint(max_leverage=3.0))
        self.risk.add_constraint(minimum_volatility_constraint(min_volatility=0.02))
        # self.risk.add_constraint(garch_constraint(max_volatility=0.03))
        # self.risk.add_constraint(portfolio_risk_constraint(max_risk=0.2))
        # self.risk.add_constraint(jump_risk_constraint(max_jump_risk=0.04))
        self.risk.add_constraint(concentration_constraint(max_concentration=0.3))
        self.risk.add_constraint(liquidity_constraint(max_position_volume=0.1))


    def execute(self, save_to_csv=True, csv_path="data/daily_top_stocks.csv"):
        """
    Mean-Reverting Strategy: Buys stocks that are below their historical mean and sells those above it.
        """
        print(f"Executing Mean-Reverting Strategy with Lookback Period {self.lookback_period}...")

        # Calculate rolling mean and standard deviation
        self.data["Rolling_Mean"] = (
            self.data.groupby("ticker")["close"]
            .rolling(window=self.lookback_period, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        self.data["Rolling_Std"] = (
            self.data.groupby("ticker")["close"]
            .rolling(window=self.lookback_period, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

        # Calculate Z-scores
        self.data["Z_Score"] = (self.data["close"] - self.data["Rolling_Mean"]) / self.data["Rolling_Std"]

        # Select stocks with the lowest Z-scores (mean reversion buy candidates)
        daily_top_stocks = (
            self.data.groupby("date", group_keys=False)
            .apply(lambda group: group.nsmallest(self.num_stocks, "Z_Score"))
            .copy()
        )

        # Shift close prices to simulate next day's returns
        daily_top_stocks["Next_Close"] = daily_top_stocks.groupby("ticker")["close"].shift(-1)
        daily_top_stocks = daily_top_stocks.dropna(subset=["Next_Close"])

        # Calculate daily returns
        daily_top_stocks["Daily_Return"] = (daily_top_stocks["Next_Close"] / daily_top_stocks["close"]) - 1

        # Allocate capital equally among selected stocks
        daily_top_stocks["Ideal_Weight"] = 1 / self.num_stocks
        daily_top_stocks["Ideal_Positions"] = (
            (self.capital * daily_top_stocks["Ideal_Weight"]) / daily_top_stocks["close"]
        )

        # Check for optimizer
        if self.optimizer is None:
            print("Optimizer not initialized. Rounding Ideal Positions to integers.")
            daily_top_stocks["Ideal_Positions"] = daily_top_stocks["Ideal_Positions"].round()

        # Calculate Ideal Capital
        daily_top_stocks["Ideal_Capital"] = daily_top_stocks["Ideal_Positions"] * daily_top_stocks["close"]

        # Use Ideal Positions as the actual positions
        self.positions = daily_top_stocks["Ideal_Positions"].values

        # Calculate portfolio value for the max leverage constraint
        daily_top_stocks["portfolio_value"] = (daily_top_stocks["Ideal_Positions"] * daily_top_stocks["close"]).sum()

        # Check constraints
        if not self.risk.check_constraints(daily_top_stocks, self.positions):
            print("Risk constraints not satisfied. Aborting execution.")
            return pd.DataFrame({
                "date": daily_top_stocks["date"].unique(),
                "Realized_Capital": [0] * len(daily_top_stocks["date"].unique()),
                "Ideal_Capital": [0] * len(daily_top_stocks["date"].unique())
            })

        # Calculate realized capital
        daily_top_stocks["Realized_Capital"] = daily_top_stocks["Ideal_Positions"] * daily_top_stocks["Next_Close"]

        # Save results
        if save_to_csv:
            daily_top_stocks.to_csv(csv_path, index=False)
            print(f"Mean-reverting stocks saved to {csv_path}")

        metrics = self.risk.evaluate_metrics(daily_top_stocks)
        # display_metrics(metrics)  # Use the formatted output

        return pd.DataFrame({
            "date": daily_top_stocks["date"].unique(),
            "Realized_Capital": daily_top_stocks.groupby("date")["Realized_Capital"].sum(),
            "Ideal_Capital": daily_top_stocks.groupby("date")["Ideal_Capital"].sum(),
        })



