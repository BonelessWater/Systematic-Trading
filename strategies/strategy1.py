from strategies.base_strategy import BaseStrategy
import numpy as np
import pandas as pd

class Strategy1(BaseStrategy):
    def execute(self, stop_loss_threshold=1.0, save_to_csv=True, csv_path="data/daily_top_stocks.csv"):
        """Selects top performers daily, optimizes positions to integer shares, and invests for the next day."""
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

        # Ideal allocation based on equal weights
        daily_top_stocks['Ideal_Weight'] = 1 / self.num_stocks  # Equal weight for each stock
        daily_top_stocks['Ideal_Positions'] = (
            (self.capital * daily_top_stocks['Ideal_Weight']) / daily_top_stocks['close']
        )

        tickers = daily_top_stocks['ticker'].unique()
        prices = daily_top_stocks.groupby('ticker')['close'].mean().reindex(tickers).values

        # Optimize positions using PortfolioOptimizer
        if self.optimizer:
            tickers = daily_top_stocks['ticker'].unique()
            ideal_positions = daily_top_stocks.groupby('ticker')['Ideal_Positions'].mean().reindex(tickers).values
            realized_positions = self.optimizer.optimize(
                curr_pos=np.zeros(len(ideal_positions)),
                ideal_pos=ideal_positions,
                prices=prices,
                learning_rate=0.3,
                iterations=1000
            )
            # Assign optimized positions back to daily_top_stocks
            ticker_to_positions = dict(zip(tickers, realized_positions))
            daily_top_stocks['Realized_Positions'] = daily_top_stocks['ticker'].map(ticker_to_positions)
        else:
            print("PortfolioOptimizer not initialized. Using ideal positions as realized positions.")
            daily_top_stocks['Realized_Positions'] = daily_top_stocks['Ideal_Positions']

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
            tracking_error = np.sqrt(np.sum((realized - ideal) ** 2))
            tracking_errors.append(tracking_error)

        # Save to CSV if required
        if save_to_csv:
            daily_top_stocks.to_csv(csv_path, index=False)
            print(f"Daily top stocks saved to {csv_path}")

        # Store results
        results = pd.DataFrame({
            'date': portfolio_realized_returns.index,
            'Realized_Capital': realized_capital.values,
            'Ideal_Capital': ideal_capital.values,
            'Tracking_Error': tracking_errors,
            'Drawdown': self.calculate_drawdown(cumulative_ideal_returns)
        })

        # Return results
        return results
