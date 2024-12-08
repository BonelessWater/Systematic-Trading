from optimizers.adam import AdamOptimizer
from optimizers.grad_descent import GradDescentOptimizer
from optimizers.greedy import GreedyOptimizer
from optimizers.sgd import StochasticGradDescentOptimizer
from optimizers.pyomo import PyomoOptimizer
from optimizers.scipy_min import ScipyOptimizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BaseStrategy:
    def __init__(self, data, risk_target, capital, num_stocks):
        """
        :param data: DataFrame with ['date', 'ticker', 'close', 'PercentChange'] columns.
        :param risk_target: Risk target parameter (for future use).
        :param capital: Total capital available for investment.
        :param num_stocks: Number of stocks to include in the portfolio.
        """
        if data is None or data.empty:
            raise ValueError("The input data is either None or empty.")

        self.data = data
        self.risk_target = risk_target
        self.capital = capital
        self.num_stocks = num_stocks
        self.window = 3  # Use a very small rolling window for faster computation

        # Ensure the data is sorted for proper processing
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        self.data = self.data.sort_values(['date', 'ticker']).dropna(subset=['date', 'ticker', 'PercentChange'])

        # Calculate covariance matrices
        print("Calculating covariance matrices...")
        self.covariance_matrices = self.calculate_covariance_matrices()

        # Initialize optimizer (can be overridden in derived classes)
        self.optimizer = None

    def calculate_covariance_matrices(self):
        """
        Calculate covariance matrices for each day in the dataset.
        Returns a dictionary with dates as keys and covariance matrices as values.
        """
        covariance_matrices = {}

        # Group by date and calculate covariance for each day
        grouped = self.data.groupby('date')
        for date, group in grouped:
            if len(group) < 2:
                # Skip days with less than 2 stocks
                continue

            # Pivot table to have tickers as columns and PercentChange as values
            pivoted = group.pivot(index='ticker', columns='date', values='PercentChange')

            # Compute rolling covariance with a small window
            if pivoted.shape[0] >= self.window:
                rolling_cov = pivoted.iloc[:, -self.window:].cov()
                covariance_matrices[date] = rolling_cov

        return covariance_matrices

    def generate_selection_csv(self, daily_top_stocks, csv_path='data/stock_selection_frequency.csv'):
        """Generate a CSV representing the proportion of times each stock is selected."""
        selection_counts = daily_top_stocks['ticker'].value_counts()
        total_days = daily_top_stocks['date'].nunique()
        selection_proportions = selection_counts / total_days

        selection_df = pd.DataFrame({
            'Ticker': selection_proportions.index,
            'Proportion_Selected': selection_proportions.values
        })
        selection_df = selection_df.sort_values(by='Proportion_Selected', ascending=False)
        selection_df.to_csv(csv_path, index=False)
        print(f"Stock selection frequency saved to {csv_path}")

    def plot_capital(self, results, tracking_errors=None):
        """
        Plots the ideal capital, realized capital, and optionally the tracking error over time.
        
        :param results: DataFrame containing 'date', 'Realized_Capital', and 'Ideal_Capital'.
        :param tracking_errors: (Optional) List or Series containing tracking error values over time.
        """
        plt.figure(figsize=(14, 8), dpi=100)

        # Add a secondary y-axis for tracking error if provided
        if tracking_errors is not None:
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(results['date'], tracking_errors, label='Tracking Error', color='red', linestyle='-.')
            ax2.set_ylabel('Tracking Error', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(False)  # Disable grid for secondary axis
            ax2.legend(loc='upper right')

        # Plot labels and legend for primary axis
        plt.title('Tracking Error Over Time')
        plt.xlabel('Date')
        plt.ylabel('Capital (USD)')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_pnl(self, result, save_path='data/pnl_plot.png', log_scale=False):
        """Plot realized and ideal capital over time, with optional logarithmic scaling."""
        # Ensure 'date' is in datetime format
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        result = result[result['date'].notna()]  # Drop rows with invalid dates

        # Check if required columns are present
        if 'Realized_Capital' not in result or 'Ideal_Capital' not in result:
            raise KeyError("Result DataFrame must contain 'Realized_Capital' and 'Ideal_Capital' columns.")

        # Plot cumulative capital over time
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(result['date'], result['Realized_Capital'], label='Realized Capital (Integer Positions)', color='blue')
        plt.plot(result['date'], result['Ideal_Capital'], label='Ideal Capital (Continuous Weights)', color='green', linestyle='--')
        
        if log_scale:
            plt.yscale('log')
            print("Logarithmic scale applied to capital.")

        plt.title('Cumulative Portfolio Capital Over Time')
        plt.xlabel('Date')
        plt.ylabel('Capital (USD)')
        plt.legend()
        plt.grid()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_capital_over_time(self, result, save_path='data/capital_over_time.png'):
        """
        Plot the total capital over time for Realized and Ideal Capital.
        """
        # Ensure the result is not empty
        if result is None or result.empty:
            print("No data available for plotting capital over time.")
            return

        # Ensure the required columns exist
        if 'Ideal_Capital' not in result or 'Realized_Capital' not in result:
            print("Missing required columns ('Ideal_Capital', 'Realized_Capital') in result.")
            return

        # Ensure 'date' is properly formatted
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        result = result[result['date'].notna()]  # Drop rows with invalid dates

        # Plot the total capital over time
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(result['date'], result['Realized_Capital'], label='Realized Capital (Integer Positions)', color='blue')
        plt.plot(result['date'], result['Ideal_Capital'], label='Ideal Capital (Continuous Weights)', color='green', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Capital (USD)')
        plt.title('Total Capital Over Time')
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"Capital graph saved to {save_path}")
        plt.show()


    def metrics(self, result):
        """Print key performance metrics including total capital, volatility"""
        # Use the final realized capital as the portfolio's performance
        final_realized_capital = result['Realized_Capital'].iloc[-1]
        initial_capital = self.capital
        total_pnl = final_realized_capital - initial_capital  # Total profit or loss

        # Calculate daily returns based on Realized Capital
        daily_returns = result['Realized_Capital'].pct_change().dropna()

        # Calculate annualized volatility
        annual_volatility = daily_returns.std() * np.sqrt(252)

        # Print key metrics
        print(f"=== Strategy Metrics ===")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Final Realized Capital: ${final_realized_capital:,.2f}")
        print(f"Annualized Volatility: {annual_volatility:.4f}")
        print(f"Max Realized Capital: ${result['Realized_Capital'].max():,.2f}")
        print(f"Min Realized Capital: ${result['Realized_Capital'].min():,.2f}")

        return {
            'Total PnL': total_pnl,
            'Final Realized Capital': final_realized_capital,
            'Annualized Volatility': annual_volatility,
            'Max Realized Capital': result['Realized_Capital'].max(),
            'Min Realized Capital': result['Realized_Capital'].min(),
        }
