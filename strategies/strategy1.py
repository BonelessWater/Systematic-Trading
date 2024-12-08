from strategies.base_strategy import BaseStrategy
from risk_management.risk_obj import Risk
from scipy.stats import norm
import pandas as pd
import numpy as np

class Strategy1Risk(Risk):
    """
    Custom Risk implementation for Strategy1.
    """
    def __init__(self, data, capital, covariance_matrices):
        # Initialize the base class with data and capital
        super().__init__(data, capital)
        self.covariance_matrices = covariance_matrices  # Add covariance matrices

    def base_constraints(self, data: dict, positions) -> bool:
        """
        Define base constraints for Strategy1.
        """
        # Constraint: Ensure remaining capital is positive
        def positive_capital_constraint(data, positions):
            remaining_capital = self.capital - (positions * data['close']).sum()
            return remaining_capital > 0  # Capital must remain positive

        # Check all base constraints
        return positive_capital_constraint(data, positions)

    def base_metrics(self, data: dict) -> dict:
        """
        Define base metrics for Strategy1.
        """
        # Metric: Value at Risk (Historical)
        def value_at_risk_historical(data):
            returns = data.get("Daily_Return", pd.Series())
            confidence_level = 0.95
            var = -returns.quantile(1 - confidence_level)
            return {"Value_at_Risk_Historical": var}

        # Metric: Portfolio standard deviation (volatility)
        def portfolio_volatility_metric(data):
            returns = data.get("Daily_Return", pd.Series())
            volatility = returns.std()
            return {"Portfolio_Volatility": volatility}

        # Evaluate and return metrics
        metrics = value_at_risk_historical(data)
        metrics.update(portfolio_volatility_metric(data))
        return metrics

class Strategy1(BaseStrategy):
    def __init__(self, data, risk_target, capital, num_stocks):
        """
        Initializes Strategy1 with a custom Risk object.
        """
        super().__init__(data, risk_target, capital, num_stocks)
        self.positions = None  # Positions will be calculated later
        self.risk = Strategy1Risk(data, capital, self.covariance_matrices)  # Pass covariance matrices

    def execute(self, save_to_csv=True, csv_path="data/daily_top_stocks.csv"):
        """
        Executes the strategy by selecting top stocks, optimizing positions, and assessing risk.
        """
        print(f"Executing Top {self.num_stocks} Winner Strategy with Integer Optimization...")

        # Sort data by date and PercentChange
        self.data = self.data.sort_values(['date', 'PercentChange'], ascending=[True, False])

        # Group by date and select the top `num_stocks` tickers with the highest PercentChange
        daily_top_stocks = (
            self.data.groupby('date', group_keys=False)
            .apply(lambda group: group.nlargest(self.num_stocks, 'PercentChange'))
            .copy()
        )

        # Shift the close prices to simulate next day's returns
        daily_top_stocks['Next_Close'] = daily_top_stocks.groupby('ticker')['close'].shift(-1)
        daily_top_stocks = daily_top_stocks.dropna(subset=['Next_Close'])  # Drop NaNs after shifting

        # Calculate daily returns for the selected stocks
        daily_top_stocks['Daily_Return'] = (daily_top_stocks['Next_Close'] / daily_top_stocks['close']) - 1

        # Equal allocation based on the number of stocks
        daily_top_stocks['Ideal_Weight'] = 1 / self.num_stocks
        daily_top_stocks['Ideal_Positions'] = (
            (self.capital * daily_top_stocks['Ideal_Weight']) / daily_top_stocks['close']
        )

        # Check for optimizer
        if self.optimizer is None:
            print("Optimizer not initialized. Rounding Ideal Positions to integers.")
            daily_top_stocks['Ideal_Positions'] = daily_top_stocks['Ideal_Positions'].round()

        # Calculate Ideal Capital
        daily_top_stocks['Ideal_Capital'] = daily_top_stocks['Ideal_Positions'] * daily_top_stocks['close']

        # Use Ideal Positions as the actual positions
        self.positions = daily_top_stocks['Ideal_Positions'].values  # Save positions for risk checking

        # Check constraints
        if not self.risk.check_constraints(daily_top_stocks, self.positions):
            print("Risk constraints not satisfied. Aborting execution.")
            # Return a DataFrame with Ideal and Realized Capital as zeros for consistency
            return pd.DataFrame({
                'date': daily_top_stocks['date'].unique(),
                'Realized_Capital': [0] * len(daily_top_stocks['date'].unique()),
                'Ideal_Capital': [0] * len(daily_top_stocks['date'].unique())
            })

        # Calculate realized capital
        daily_top_stocks['Realized_Capital'] = daily_top_stocks['Ideal_Positions'] * daily_top_stocks['Next_Close']

        # Return results
        results = pd.DataFrame({
            'date': daily_top_stocks['date'].unique(),
            'Realized_Capital': daily_top_stocks.groupby('date')['Realized_Capital'].sum(),
            'Ideal_Capital': daily_top_stocks.groupby('date')['Ideal_Capital'].sum(),
        })

        if save_to_csv:
            daily_top_stocks.to_csv(csv_path, index=False)
            print(f"Daily top stocks saved to {csv_path}")

        return results
