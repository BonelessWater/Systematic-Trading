import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Strategy1:
    def __init__(self, data, risk_target, capital):
        if 'PercentChange' not in data.columns:
            raise ValueError("Data must contain 'PercentChange' column.")
        self.data = data
        self.risk_target = risk_target
        self.capital = capital

    def execute(self):
        """Simple moving average crossover strategy."""
        print("Executing Strategy1...")
        data = self.data.copy()

        # Calculate moving averages
        data['SMA_10'] = data['close'].rolling(window=10).mean()
        data['SMA_30'] = data['close'].rolling(window=30).mean()

        # Generate buy/sell signals
        data['Signal'] = np.where(data['SMA_10'] > data['SMA_30'], 1, 0)
        data['Position'] = data['Signal'].diff()

        # Calculate strategy returns and PnL
        data['Strategy_Returns'] = data['PercentChange'] * data['Position'].shift(1)
        data['PnL'] = self.capital * data['Strategy_Returns'].cumsum()

        # Calculate realized volatility (annualized)
        data['Volatility'] = data['Strategy_Returns'].rolling(window=30).std() * np.sqrt(252)

        # Reset index to access 'date' as a column
        data = data.reset_index()

        return data[['date', 'close', 'PnL', 'Volatility']].set_index('date')

    def plot_pnl(self, result):
        """Plot cumulative PnL over time."""
        plt.figure(figsize=(10, 6))
        result['PnL'].plot(title='Cumulative PnL - Strategy 1', xlabel='Date', ylabel='PnL (USD)')
        plt.grid(True)
        plt.show()

    def metrics(self, result):
        """Print performance metrics."""
        total_pnl = result['PnL'].iloc[-1]
        annual_volatility = result['Volatility'].mean() * np.sqrt(252)  # Annualized volatility
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Annualized Volatility: {annual_volatility:.4f}")


class Strategy2:
    def __init__(self, data, risk_target, capital):
        if 'PercentChange' not in data.columns:
            raise ValueError("Data must contain 'PercentChange' column.")
        self.data = data
        self.risk_target = risk_target
        self.capital = capital

    def execute(self):
        """Momentum strategy based on RSI."""
        print("Executing Strategy2...")
        data = self.data.copy()

        # Calculate RSI
        delta = data['close'].diff(1)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()

        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Generate buy/sell signals based on RSI thresholds
        data['Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
        data['Position'] = data['Signal'].diff()

        # Calculate strategy returns and PnL
        data['Strategy_Returns'] = data['PercentChange'] * data['Position'].shift(1)
        data['PnL'] = self.capital * data['Strategy_Returns'].cumsum()

        # Calculate realized volatility (annualized)
        data['Volatility'] = data['Strategy_Returns'].rolling(window=30).std() * np.sqrt(252)

        # Reset index to access 'date' as a column
        data = data.reset_index()

        return data[['date', 'close', 'PnL', 'Volatility']].set_index('date')

    def plot_pnl(self, result):
        """Plot cumulative PnL over time."""
        plt.figure(figsize=(10, 6))
        result['PnL'].plot(title='Cumulative PnL - Strategy 2', xlabel='Date', ylabel='PnL (USD)')
        plt.grid(True)
        plt.show()

    def metrics(self, result):
        """Print performance metrics."""
        total_pnl = result['PnL'].iloc[-1]
        annual_volatility = result['Volatility'].mean() * np.sqrt(252)  # Annualized volatility
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Annualized Volatility: {annual_volatility:.4f}")
