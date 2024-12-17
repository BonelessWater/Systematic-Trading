import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_metrics(data):
    # Daily returns
    returns = data.pct_change().dropna()
    
    # Portfolio weights (equal weighting)
    num_stocks = len(data.columns)
    weights = np.array([1 / num_stocks] * num_stocks)

    # Compounded Annual Growth Rate (CAGR)
    total_return = (data.iloc[-1] / data.iloc[0]) - 1
    cagr = (1 + total_return.mean()) ** (252 / len(returns)) - 1

    # Annualized Returns
    annualized_return = returns.mean().dot(weights) * 252

    # Covariance Matrix and Portfolio Volatility
    cov_matrix = returns.cov() * 252
    portfolio_volatility = np.sqrt(weights @ cov_matrix @ weights)

    # Sharpe Ratio
    risk_free_rate = 0.02  # Example risk-free rate
    sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_volatility

    # Sortino Ratio
    downside_returns = returns[returns < 0].mean().dot(weights) * 252
    sortino_ratio = (annualized_return - risk_free_rate) / abs(downside_returns)

    # Profit Factor
    gains = returns[returns > 0].sum().dot(weights)
    losses = abs(returns[returns < 0].sum().dot(weights))
    profit_factor = gains / losses

    # Win Rate
    win_rate = (returns > 0).sum().sum() / returns.size

    # Maximum Drawdown
    cumulative_returns = (1 + returns.sum(axis=1)).cumprod()
    max_drawdown = ((cumulative_returns / cumulative_returns.cummax()) - 1).min()

    # Risk Metrics
    beta = (returns.cov().iloc[:, 0] / returns.var().iloc[0]).mean()  # Example Beta calculation
    value_at_risk = np.percentile(returns.sum(axis=1), 5)  # 5% VaR
    tail_risk = returns.sum(axis=1)[returns.sum(axis=1) <= value_at_risk].mean()

    metrics = {
        'CAGR': cagr,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Profit Factor': profit_factor,
        'Win Rate': win_rate,
        'Max Drawdown': max_drawdown,
        'Portfolio Volatility': portfolio_volatility,
        'Beta': beta,
        'Value at Risk (5%)': value_at_risk,
        'Tail Risk': tail_risk
    }
    return metrics

def position_sizing(capital, risk_per_trade, stop_loss_pct):
    # Position size based on risk management
    return (capital * risk_per_trade) / stop_loss_pct

def display_metrics(metrics):
    print("Portfolio Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2%}" if isinstance(value, float) else f"{key}: {value}")

def plot_equal_weighted_portfolio(data):
    daily_returns = data.pct_change().dropna()

    # Compute the equal-weighted portfolio returns
    num_assets = data.shape[1]
    equal_weighted_returns = daily_returns.mean(axis=1)

    # Compute cumulative returns for the equal-weighted portfolio
    cumulative_returns = (1 + equal_weighted_returns).cumprod()

    # Plot the equal-weighted portfolio value over time
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="Equal-Weighted Portfolio")
    plt.title("Equal-Weighted Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()

def main():
    tickers = ['IONQ', 'RGTI', 'QUBT', 'QBTS', 'QMCO', 'QS', 'ARQQ', 'QSI']  # Quantum computing companies
    start_date = '2024-06-01'
    end_date = '2024-12-16'

    # Download data
    data = download_data(tickers, start_date, end_date)

    # Plot portfolio weights
    plot_equal_weighted_portfolio(data)

    # Calculate metrics
    metrics = calculate_metrics(data)

    # Display metrics
    display_metrics(metrics)

    # Example position sizing
    capital = 1000  # Example capital
    risk_per_trade = 0.01  # Risk per trade
    stop_loss_pct = 0.05  # Stop loss percentage
    position_size = position_sizing(capital, risk_per_trade, stop_loss_pct)
    print(f"Position Size: ${position_size:.2f}")

if __name__ == "__main__":
    main()
