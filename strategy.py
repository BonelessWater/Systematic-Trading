from port_opt.grad_descent import PortfolioOptimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Strategy1:
    def __init__(self, data, risk_target, capital, num_stocks=100, correlation_matrix=None):
        """
        :param data: DataFrame with ['date', 'ticker', 'close', 'PercentChange'] columns.
        :param risk_target: Risk target parameter (for future use).
        :param capital: Total capital available for investment.
        :param num_stocks: Number of stocks to include in the portfolio.
        :param correlation_matrix: Correlation matrix for assets used in optimization.
        """
        if data is None or data.empty:
            raise ValueError("The input data is either None or empty.")

        self.data = data
        self.risk_target = risk_target
        self.capital = capital
        self.num_stocks = num_stocks
        self.optimizer = PortfolioOptimizer(correlation_matrix) if correlation_matrix else None

    def execute(self, stop_loss_threshold=1.0):
        """Selects top performers daily, optimizes positions to integer shares, and invests for the next day."""
        print(f"Executing Top {self.num_stocks} Winner Strategy with Integer Optimization...")

        # Sort data by date and PercentChange
        self.data = self.data.sort_values(['date', 'PercentChange'], ascending=[True, False])

        # Debugging: Check if 'ticker' column exists
        print("Data Columns:", self.data.columns)

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

        # Ideal allocation based on equal weights
        daily_top_stocks['Ideal_Weight'] = 1 / self.num_stocks  # Equal weight for each stock
        daily_top_stocks['Ideal_Positions'] = (
            (self.capital * daily_top_stocks['Ideal_Weight']) / daily_top_stocks['close']
        )

        # Optimize positions using PortfolioOptimizer
        if self.optimizer:
            tickers = daily_top_stocks['ticker'].unique()
            ideal_positions = daily_top_stocks.groupby('ticker')['Ideal_Positions'].mean().reindex(tickers).values
            realized_positions = self.optimizer.gradient_descent(
                curr_pos=np.zeros(len(ideal_positions)),
                ideal_pos=ideal_positions,
                learning_rate=3,
                iterations=100
            )
        else:
            print("PortfolioOptimizer not initialized. Using ideal positions as realized positions.")
            realized_positions = daily_top_stocks['Ideal_Positions']

        # Map integer positions back to tickers
        ticker_to_realized = dict(zip(daily_top_stocks['ticker'].unique(), realized_positions))
        daily_top_stocks['Realized_Positions'] = daily_top_stocks['ticker'].map(ticker_to_realized)

        # Calculate portfolio daily returns for realized and ideal capital
        daily_top_stocks['Realized_Return'] = (
            daily_top_stocks['Realized_Positions'] * daily_top_stocks['Daily_Return']
        )
        daily_top_stocks['Ideal_Return'] = (
            daily_top_stocks['Ideal_Positions'] * daily_top_stocks['Daily_Return']
        )
        portfolio_realized_returns = daily_top_stocks.groupby('date')['Realized_Return'].sum() / self.capital
        portfolio_ideal_returns = daily_top_stocks.groupby('date')['Ideal_Return'].sum() / self.capital

        # Calculate cumulative returns and capital
        cumulative_realized_returns = (1 + portfolio_realized_returns).cumprod()
        cumulative_ideal_returns = (1 + portfolio_ideal_returns).cumprod()
        realized_capital = self.capital * cumulative_realized_returns
        ideal_capital = self.capital * cumulative_ideal_returns

        # Calculate tracking error over time
        tracking_errors = []
        for date, group in daily_top_stocks.groupby('date'):
            realized = group['Realized_Positions'].values
            ideal = group['Ideal_Positions'].values
            tracking_errors.append(np.sqrt(np.sum((realized - ideal) ** 2)))

        # Store results
        results = pd.DataFrame({
            'date': portfolio_realized_returns.index,
            'Realized_Capital': realized_capital.values,
            'Ideal_Capital': ideal_capital.values,
            'Tracking_Error': tracking_errors
        })

        # Calculate Drawdown for Realized Capital
        cumulative_max = results['Realized_Capital'].cummax()
        results['Drawdown'] = (results['Realized_Capital'] - cumulative_max) / cumulative_max
        
        # Plot capital and tracking error
        self.plot_capital(results, tracking_errors=results['Tracking_Error'])

        # Debugging: Print a summary
        print(f"Final Realized Capital: ${realized_capital.iloc[-1]:,.2f}")
        print(f"Final Ideal Capital: ${ideal_capital.iloc[-1]:,.2f}")

        return results


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
        # Ensure 'date' is in datetime format
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        
        # Drop NaNs and sort by date
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
