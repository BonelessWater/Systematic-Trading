import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        """Selects top performers daily and invests for the next day, with stop-loss to terminate early."""
        print(f"Executing Top {self.num_stocks} Winner Strategy...")
        self.data = self.data.sort_values(['date', 'PercentChange'], ascending=[True, False])

        # Debugging: Check if 'ticker' column exists
        print("Data Columns:", self.data.columns)

        # Group by date and select the top `num_stocks` tickers with the highest PercentChange
        daily_top_stocks = self.data.groupby('date').head(self.num_stocks)

        # Ensure it’s a copy to avoid warnings
        daily_top_stocks = daily_top_stocks.copy()

        # Shift the close prices to simulate next day's returns
        daily_top_stocks.loc[:, 'Next_Close'] = daily_top_stocks.groupby('ticker')['close'].shift(-1)
        daily_top_stocks = daily_top_stocks.dropna(subset=['Next_Close'])  # Drop NaNs after shifting

        # Calculate daily returns for the selected stocks
        daily_top_stocks['Daily_Return'] = (daily_top_stocks['Next_Close'] / daily_top_stocks['close']) - 1

        # Calculate equal allocation and portfolio daily return
        daily_top_stocks['Weight'] = 1 / self.num_stocks  # Adjust weight based on the number of stocks
        daily_top_stocks['Weighted_Return'] = daily_top_stocks['Weight'] * daily_top_stocks['Daily_Return']
        portfolio_returns = daily_top_stocks.groupby('date')['Weighted_Return'].sum()

        # Calculate cumulative returns and cumulative PnL
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        cumulative_pnl = self.capital * cumulative_returns

        # Store results with drawdown
        result = pd.DataFrame({
            'date': cumulative_pnl.index,
            'PnL': cumulative_pnl.values,
            'Cumulative_Returns': cumulative_returns[:len(cumulative_pnl)].values
        })

        # Ensure dates are valid and sorted
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        result = result.dropna(subset=['date']).sort_values(by='date')

        result['Drawdown'] = self.calculate_drawdown(cumulative_returns)
        # Create the frequency table of stock selections
        self.generate_selection_csv(daily_top_stocks)

        return result
    
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
