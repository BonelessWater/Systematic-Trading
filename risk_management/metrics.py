import numpy as np
from scipy.stats import norm
from typing import Callable, Any, List, Dict
import pandas as pd

def portfolio_volatility_metric():
    """
    Calculates the portfolio's overall volatility, which is the standard deviation of daily returns.
    This metric measures the risk or variability of the portfolio's returns over time.
    """
    def metric(data: Dict[str, Any]) -> Dict[str, Any]:
        returns = data.get("Daily_Return", pd.Series(dtype=float))
        if returns.empty:
            print("Portfolio Volatility Metric: No returns data available.")
            return {"Portfolio_Volatility": None}

        volatility = returns.std()
        print(f"Calculated Portfolio Volatility: {volatility:.4f}")
        return {"Portfolio_Volatility": volatility}

    return metric

def sharpe_ratio_metric(risk_free_rate: float = 0.02):
    """
    Calculates the Sharpe Ratio, which measures the portfolio's risk-adjusted return.
    The Sharpe Ratio is calculated as the average excess return per unit of volatility.
    """
    def metric(data: Dict[str, Any]) -> Dict[str, Any]:
        returns = data.get("Daily_Return", pd.Series(dtype=float))
        if returns.empty:
            print("Sharpe Ratio Metric: No returns data available.")
            return {"Sharpe_Ratio": None}

        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / returns.std()
        print(f"Calculated Sharpe Ratio: {sharpe_ratio:.4f}")
        return {"Sharpe_Ratio": sharpe_ratio}

    return metric

def sortino_ratio_metric(risk_free_rate: float = 0.01):
    """
    Calculates the Sortino Ratio, a variation of the Sharpe Ratio that only penalizes downside risk.
    This metric evaluates the portfolio's risk-adjusted return, focusing solely on negative deviations.
    """
    def metric(data: Dict[str, Any]) -> Dict[str, Any]:
        returns = data.get("Daily_Return", pd.Series(dtype=float))
        if returns.empty:
            print("Sortino Ratio Metric: No returns data available.")
            return {"Sortino_Ratio": None}

        downside_risk = np.sqrt(np.mean(np.minimum(0, returns - risk_free_rate) ** 2))
        excess_returns = returns.mean() - risk_free_rate / 252
        sortino_ratio = excess_returns / downside_risk if downside_risk != 0 else None
        print(f"Calculated Sortino Ratio: {sortino_ratio:.4f}")
        return {"Sortino_Ratio": sortino_ratio}

    return metric

def value_at_risk_metric(confidence_level: float = 0.95):
    """
    Calculates the Value at Risk (VaR), which quantifies the maximum expected loss over a given time frame
    at a specified confidence level. VaR provides a probabilistic estimate of potential downside risk.
    """
    def metric(data: Dict[str, Any]) -> Dict[str, Any]:
        returns = data.get("Daily_Return", pd.Series(dtype=float))
        if returns.empty:
            print("Value at Risk Metric: No returns data available.")
            return {"Value_at_Risk": None}

        var = -returns.quantile(1 - confidence_level)
        print(f"Calculated Value at Risk (VaR) at {confidence_level*100:.1f}%: {var:.4f}")
        return {"Value_at_Risk": var}

    return metric

def conditional_value_at_risk_metric(confidence_level: float = 0.95):
    """
    Calculates the Conditional Value at Risk (CVaR), also known as Expected Shortfall.
    CVaR estimates the average loss in the worst-case scenario beyond the Value at Risk (VaR) threshold.
    """
    def metric(data: Dict[str, Any]) -> Dict[str, Any]:
        returns = data.get("Daily_Return", pd.Series(dtype=float))
        if returns.empty:
            print("CVaR Metric: No returns data available.")
            return {"Conditional_Value_at_Risk": None}

        var = -returns.quantile(1 - confidence_level)
        cvar = -returns[returns <= -var].mean()
        print(f"Calculated CVaR at {confidence_level*100:.1f}%: {cvar:.4f}")
        return {"Conditional_Value_at_Risk": cvar}

    return metric

def beta_metric():
    """
    Calculates Beta, which measures the sensitivity of the portfolio's returns relative to the market.
    Beta indicates how much the portfolio's returns move compared to the overall market's returns.
    """
    def metric(data: Dict[str, Any]) -> Dict[str, Any]:
        # Retrieve returns from the data
        returns = data.get("Daily_Return", pd.Series(dtype=float))
        
        if returns.empty or len(returns) <= 1:
            print("Beta Metric: Insufficient returns data.")
            return {"Beta": None}

        # Calculate Beta using self-covariance and variance
        beta = np.cov(returns, returns)[0, 1] / np.var(returns) if np.var(returns) != 0 else 0
        print(f"Calculated Beta: {beta:.4f}")
        return {"Beta": beta}

    return metric

def greek_values_metric(
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.02,
    volatility: float = 0.2
):
    """
    Calculates Delta, Gamma, and Vega, the key Greek values for options pricing.
    - Delta measures the sensitivity of the option's price to changes in the underlying price.
    - Gamma measures the sensitivity of Delta to changes in the underlying price.
    - Vega measures the sensitivity of the option's price to changes in volatility.
    These metrics are derived using the Black-Scholes model.
    """
    def metric(data: Dict[str, Any]) -> Dict[str, Any]:
        # Validate input parameters
        if underlying_price <= 0 or strike_price <= 0 or time_to_expiry <= 0:
            print("Greek Values Metric: Invalid input parameters.")
            return {"Delta": None, "Gamma": None, "Vega": None}

        # Calculate d1 and d2 for Black-Scholes model
        d1 = (np.log(underlying_price / strike_price) +
              (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        # Calculate Delta, Gamma, and Vega
        delta = norm.cdf(d1)  # N(d1)
        gamma = norm.pdf(d1) / (underlying_price * volatility * np.sqrt(time_to_expiry))  # N'(d1) / (S * σ * sqrt(T))
        vega = underlying_price * norm.pdf(d1) * np.sqrt(time_to_expiry)  # S * N'(d1) * sqrt(T)

        print(f"Calculated Greeks - Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: {vega:.4f}")
        return {"Delta": delta, "Gamma": gamma, "Vega": vega}

    return metric

def upside_potential_ratio_metric():
    """
    Calculates the Upside Potential Ratio, the portfolio's positive deviation divided by its downside risk.
    This metric evaluates the balance of upside and downside risks.
    """
    def metric(data: Dict[str, Any]) -> Dict[str, Any]:
        returns = data.get("Daily_Return", pd.Series(dtype=float))
        if returns.empty:
            print("Upside Potential Ratio Metric: No returns data available.")
            return {"Upside_Potential_Ratio": None}

        upside_potential = np.mean(returns[returns > 0])
        downside_risk = np.sqrt(np.mean(np.minimum(0, returns) ** 2))
        upside_potential_ratio = upside_potential / downside_risk if downside_risk != 0 else None
        print(f"Calculated Upside Potential Ratio: {upside_potential_ratio:.6f}")
        return {"Upside_Potential_Ratio": upside_potential_ratio}

    return metric
