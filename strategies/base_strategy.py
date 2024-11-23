from optimizers.grad_descent import GradDescentOptimizer
from optimizers.adam import AdamOptimizer
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
        :param window: Rolling window size for covariance calculation.
        """
        if data is None or data.empty:
            raise ValueError("The input data is either None or empty.")

        self.data = data
        self.risk_target = risk_target
        self.capital = capital
        self.num_stocks = num_stocks
        self.window = 30

        # Ensure the data is sorted for proper processing
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        self.data = self.data.sort_values(['date', 'ticker']).dropna(subset=['date', 'ticker', 'PercentChange'])

        # Calculate covariance matrices
        print("Calculating covariance matrices...")
        self.covariance_matrices = self.calculate_covariance_matrices()

        # Initialize optimizer using the most recent covariance matrix
        if self.covariance_matrices:
            last_date = max(self.covariance_matrices.keys())
            correlation_matrix = self.covariance_matrices[last_date]
            self.optimizer = GradDescentOptimizer(correlation_matrix, capital)
            print(f"Optimizer initialized with correlation matrix for {last_date}.")
        else:
            self.optimizer = None
            print("No valid covariance matrices calculated. Optimizer not initialized.")

    def calculate_covariance_matrices(self):
        """
        Calculate rolling covariance matrices for each day.

        :return: Dictionary mapping each date to its covariance matrix.
        """
        unique_dates = self.data['date'].unique()
        covariance_matrices = {}

        for date in unique_dates:
            # Filter data for the rolling window
            start_date = date - pd.Timedelta(days=self.window)
            rolling_data = self.data[(self.data['date'] > start_date) & (self.data['date'] <= date)]

            # Ensure there are no duplicate rows for pivot
            rolling_data = rolling_data.drop_duplicates(subset=['date', 'ticker'])

            # Pivot the data to get tickers as columns
            pivoted_data = rolling_data.pivot(index='date', columns='ticker', values='PercentChange')

            # Drop columns with NaN values (tickers not present in the window)
            pivoted_data = pivoted_data.dropna(axis=1, how='any')

            # Calculate covariance matrix
            if not pivoted_data.empty:
                covariance_matrix = pivoted_data.cov().values
                covariance_matrices[date] = covariance_matrix

        if not covariance_matrices:
            print("No covariance matrices were calculated. Ensure data has sufficient coverage.")
        return covariance_matrices

    def calculate_drawdown(self, cumulative_returns):
        """Calculate the drawdown from the cumulative returns series."""
        cumulative_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - cumulative_max) / (cumulative_max + 1e-8)  # Add epsilon to avoid divide-by-zero
        return drawdown

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
        
    def calculate_drawdown(self, cumulative_returns):
        """Calculate the drawdown from the cumulative returns series."""
        cumulative_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - cumulative_max) / (cumulative_max + 1e-8)  # Add epsilon to avoid divide-by-zero
        return drawdown

    def plot_drawdown(self, result):
        """Plot the drawdown over time with enhanced checks for data validity."""
        # Ensure 'date' is in datetime format and handle index-column ambiguity
        if 'date' in result.index.names:
            result = result.reset_index(drop=True)
        result['date'] = pd.to_datetime(result['date'], errors='coerce')

        # Drop NaNs and sort by 'date'
        result = result.dropna(subset=['date', 'Drawdown']).sort_values(by='date')

        # Debugging output for validation
        print("Result DataFrame after cleaning:")
        print(result.head())

        # Check for valid drawdown data
        if result['Drawdown'].isna().all():
            print("Drawdown column contains all NaN values. Check the drawdown calculation.")
            return

        if result.empty:
            print("No valid data to plot.")
            return

        # Plot the drawdown
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(result['date'], result['Drawdown'], label='Drawdown', color='tab:red')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.title('Drawdown Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

    def largest_drawdown(self, capital_series_or_result):
        """
        Calculate the largest drawdown from a capital series or a DataFrame with a 'Drawdown' column.
        :param capital_series_or_result: Either a pandas Series (capital over time) or DataFrame with 'Drawdown'.
        :return: The largest drawdown value.
        """
        if isinstance(capital_series_or_result, pd.DataFrame) and 'Drawdown' in capital_series_or_result:
            # Use the precomputed 'Drawdown' column
            return capital_series_or_result['Drawdown'].min()
        elif isinstance(capital_series_or_result, pd.Series):
            # Calculate drawdown directly from a capital series
            cumulative_max = capital_series_or_result.cummax()
            drawdown = (capital_series_or_result - cumulative_max) / cumulative_max
            return drawdown.min()
        else:
            raise ValueError("Input must be a pandas Series (capital) or DataFrame with 'Drawdown'.")

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
        # Ensure 'date' is properly formatted
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        result = result[result['date'].notna()]  # Drop rows with invalid dates

        # Check if required columns are present
        if 'Realized_Capital' not in result or 'Ideal_Capital' not in result:
            raise KeyError("Result DataFrame must contain 'Realized_Capital' and 'Ideal_Capital' columns.")

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
        """Print key performance metrics including total capital, volatility, and largest drawdown."""
        # Use the final realized capital as the portfolio's performance
        final_realized_capital = result['Realized_Capital'].iloc[-1]
        initial_capital = self.capital
        total_pnl = final_realized_capital - initial_capital  # Total profit or loss

        # Calculate daily returns based on Realized Capital
        daily_returns = result['Realized_Capital'].pct_change().dropna()

        # Calculate largest drawdown
        max_drawdown = self.largest_drawdown(result)

        # Calculate annualized volatility
        annual_volatility = daily_returns.std() * np.sqrt(252)

        # Print key metrics
        print(f"=== Strategy Metrics ===")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Final Realized Capital: ${final_realized_capital:,.2f}")
        print(f"Largest Drawdown: {max_drawdown:.4%}")
        print(f"Annualized Volatility: {annual_volatility:.4f}")
        print(f"Max Realized Capital: ${result['Realized_Capital'].max():,.2f}")
        print(f"Min Realized Capital: ${result['Realized_Capital'].min():,.2f}")

        return {
            'Total PnL': total_pnl,
            'Final Realized Capital': final_realized_capital,
            'Largest Drawdown': max_drawdown,
            'Annualized Volatility': annual_volatility,
            'Max Realized Capital': result['Realized_Capital'].max(),
            'Min Realized Capital': result['Realized_Capital'].min(),
        }
