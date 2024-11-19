from port_opt import gradient_descent, objective_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Strategy1:
    def __init__(self, data, risk_target, capital, num_stocks=100):
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
        self.num_stocks = num_stocks  # Number of top stocks to select

    def execute(self, stop_loss_threshold=1.0):
        """Executes strategy using gradient descent and plots comparisons."""
        print(f"Executing Top {self.num_stocks} Winner Strategy with Gradient Descent...")

        # Ensure required columns are present
        required_columns = {'date', 'PercentChange', 'ticker', 'close'}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")

        # Sort data by date and PercentChange
        self.data = self.data.sort_values(['date', 'PercentChange'], ascending=[True, False])

        # Group by date and select the top `num_stocks` with the highest PercentChange
        daily_top_stocks = (
            self.data.groupby('date', group_keys=False)
            .apply(lambda group: group.nlargest(self.num_stocks, 'PercentChange'))
            .copy()
        )

        # Calculate ideal positions (mean PercentChange for each ticker)
        tickers = daily_top_stocks['ticker'].unique()
        ideal_positions = (
            daily_top_stocks.groupby('ticker')['PercentChange'].mean().reindex(tickers).values
        )

        # Compute the correlation matrix for the ideal positions
        correlation_matrix = np.corrcoef(np.tile(ideal_positions, (len(ideal_positions), 1)))

        # Initialize current positions to zeros
        current_positions = np.zeros(len(ideal_positions), dtype=np.float64)

        # Lists to store results
        realized_capital = [self.capital]
        benchmark_capital = [self.capital]
        tracking_errors = []

        # Iterate through each date
        for date, group in daily_top_stocks.groupby('date'):
            # Get realized positions using gradient descent
            realized_positions = gradient_descent(current_positions, ideal_positions, correlation_matrix)
            current_positions = realized_positions  # Update positions for next step

            # Normalize realized positions to form weights
            total_realized = np.sum(realized_positions)
            realized_weights = realized_positions / total_realized if total_realized > 0 else np.zeros_like(realized_positions)

            # Normalize ideal positions to use as benchmark weights
            total_ideal = np.sum(ideal_positions)
            normalized_positions = ideal_positions / total_ideal if total_ideal > 0 else np.zeros_like(ideal_positions)

            # Map positions to tickers
            ticker_to_realized = dict(zip(tickers, realized_weights))
            ticker_to_normalized = dict(zip(tickers, normalized_positions))

            group['Realized_Weights'] = group['ticker'].map(ticker_to_realized)
            group['Normalized_Weights'] = group['ticker'].map(ticker_to_normalized)

            # Shift close prices to simulate next day's returns
            group['Next_Close'] = group.groupby('ticker')['close'].shift(-1)
            group = group.dropna(subset=['Next_Close'])  # Drop rows with NaN Next_Close
            group['Daily_Return'] = (group['Next_Close'] / group['close']) - 1

            # Realized strategy capital
            group['Realized_Contribution'] = group['Realized_Weights'] * group['Daily_Return']
            realized_capital.append(realized_capital[-1] * (1 + group['Realized_Contribution'].sum()))

            # Benchmark capital
            group['Benchmark_Contribution'] = group['Normalized_Weights'] * group['Daily_Return']
            benchmark_capital.append(benchmark_capital[-1] * (1 + group['Benchmark_Contribution'].sum()))

            # Tracking error
            tracking_error = np.sqrt(np.sum((group['Realized_Weights'] - group['Normalized_Weights']) ** 2))
            tracking_errors.append(tracking_error)

        # Create DataFrame for results
        dates = daily_top_stocks['date'].unique()
        results = pd.DataFrame({
            'date': dates,
            'Realized_Capital': realized_capital[1:],  # Exclude initial capital
            'Benchmark_Capital': benchmark_capital[1:],
            'Tracking_Error': tracking_errors,
        })

        # Plot results
        self.plot_results(results)

        # Print summary metrics
        print(f"Final Realized Capital: ${realized_capital[-1]:,.2f}")
        print(f"Final Benchmark Capital: ${benchmark_capital[-1]:,.2f}")
        print(f"Average Tracking Error: {np.mean(tracking_errors):.4f}")

        return results

    def plot_results(self, results):
        """Plots the realized vs benchmark capital and tracking error over time."""
        plt.figure(figsize=(14, 7), dpi=100)

        # Plot capital
        plt.subplot(2, 1, 1)
        plt.plot(results['date'], results['Realized_Capital'], label='Realized Capital', color='blue')
        plt.plot(results['date'], results['Benchmark_Capital'], label='Benchmark Capital', color='green')
        plt.title('Portfolio Capital Over Time')
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.legend()
        plt.grid()

        # Plot tracking error
        plt.subplot(2, 1, 2)
        plt.plot(results['date'], results['Tracking_Error'], label='Tracking Error', color='red')
        plt.title('Tracking Error Over Time')
        plt.xlabel('Date')
        plt.ylabel('Tracking Error')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()


    def generate_selection_csv(self, daily_top_stocks, csv_path='data/stock_selection_frequency.csv'):
        """Generate a CSV representing the proportion of times each stock is selected."""
        # Debugging: Check columns of daily_top_stocks
        print("Daily Top Stocks Columns:", daily_top_stocks.columns)

        # Ensure 'ticker' column exists before value_counts()
        if 'ticker' not in daily_top_stocks.columns:
            raise KeyError("The 'ticker' column is missing from the DataFrame.")

        # Count the occurrences of each stock in the top selections
        selection_counts = daily_top_stocks['ticker'].value_counts()

        # Calculate the proportion of times each stock was selected
        total_days = daily_top_stocks['date'].nunique()  # Total number of unique trading days
        selection_proportions = selection_counts / total_days

        # Create a DataFrame to store the results
        selection_df = pd.DataFrame({
            'Ticker': selection_proportions.index,
            'Proportion_Selected': selection_proportions.values
        })

        # Sort by proportion in descending order
        selection_df = selection_df.sort_values(by='Proportion_Selected', ascending=False)

        # Save to CSV
        selection_df.to_csv(csv_path, index=False)
        print(f"Stock selection frequency saved to {csv_path}")

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

    def largest_drawdown(self, result):
        """Calculate the largest drawdown."""
        return result['Drawdown'].min()  # Most negative drawdown is the largest

    def plot_pnl(self, result, csv_path='data/pnl_drawdown.csv', save_path='data/pnl_plot.png', log_scale=False):
        """Plot cumulative PnL and drawdown over time, with optional saving to a CSV and logarithmic scaling."""
        # Data cleaning
        result = result.dropna(subset=['date', 'PnL', 'Drawdown'])
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        result = result[result['date'].notna()]
        result = result.sort_values(by='date')

        # Check for empty DataFrame
        if result.empty:
            print("No valid data to plot.")
            return

        # Plot cumulative PnL and drawdown
        fig, ax1 = plt.subplots(figsize=(12, 6), dpi=100)

        ax1.set_xlabel('Date')
        ax1.set_ylabel('PnL (USD)', color='tab:blue')
        ax1.plot(result['date'], result['PnL'], label='Cumulative PnL', color='tab:blue')
        if log_scale and (result['PnL'] > 0).all():
            ax1.set_yscale('log')
            print("Logarithmic scale applied to PnL.")
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Drawdown', color='tab:red')
        ax2.plot(result['date'], result['Drawdown'], label='Drawdown', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title('Cumulative PnL and Drawdown Over Time')
        fig.tight_layout()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_capital_over_time(self, result, save_path='capital_over_time.png'):
        """Plot the total capital over time based on PnL."""
        result = result.dropna(subset=['date', 'PnL'])
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        result = result[result['date'].notna()]

        # Calculate total capital over time
        result['Total_Capital'] = self.capital + result['PnL']

        # Plot the total capital over time
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(result['date'], result['Total_Capital'], label='Total Capital', color='green')
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
        """Print key performance metrics including total PnL, volatility, and largest drawdown."""
        total_pnl = result['PnL'].iloc[-1]  # Final cumulative PnL
        max_drawdown = self.largest_drawdown(result)  # Largest drawdown from the series
        daily_returns = result['Cumulative_Returns'].pct_change().dropna()  # Daily percentage returns

        # Calculate annualized volatility from daily returns
        annual_volatility = daily_returns.std() * np.sqrt(252)

        # Print key metrics clearly
        print(f"=== Strategy Metrics ===")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Largest Drawdown: {max_drawdown:.4%}")
        print(f"Annualized Volatility: {annual_volatility:.4f}")
        print(f"Max PnL: ${result['PnL'].max():,.2f}")
        print(f"Min PnL: ${result['PnL'].min():,.2f}")

        return {
            'Total PnL': total_pnl,
            'Largest Drawdown': max_drawdown,
            'Annualized Volatility': annual_volatility,
            'Max PnL': result['PnL'].max(),
            'Min PnL': result['PnL'].min()
        }
