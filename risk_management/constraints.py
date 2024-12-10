import numpy as np
from typing import Callable, Any, List, Dict
from arch import arch_model
import pandas as pd

def positive_capital_constraint():
    """
    Ensures the portfolio does not exceed the available capital.
    The constraint checks that the remaining capital after accounting for positions is non-negative.
    """
    def constraint(data: Dict[str, Any], positions: np.ndarray, capital: float) -> bool:
        remaining_capital = capital - np.dot(positions, data['close'].values)
        if remaining_capital < 0:
            print(f"Positive capital constraint failed. Remaining capital: {remaining_capital:.2f}")
            return False
        return True
    return constraint


def max_leverage_constraint(max_leverage: float):
    """
    Ensures that the portfolio's leverage (positions relative to portfolio value) does not exceed the specified limit.
    Leverage is calculated as the total value of positions divided by the portfolio value.
    """
    def constraint(data: Dict[str, Any], positions: np.ndarray, capital: float) -> bool:
        portfolio_value = data.get('portfolio_value', None)
        if portfolio_value is None:
            print("Max leverage constraint failed. Portfolio value is missing.")
            return False
        if isinstance(portfolio_value, (pd.Series, np.ndarray)):
            portfolio_value = portfolio_value.sum()

        leverage = np.sum(positions * data['close'].values) / portfolio_value
        print(f"Calculated leverage: {leverage:.2f}, Allowed max leverage: {max_leverage:.2f}")

        if leverage > max_leverage:
            print(f"Max leverage constraint failed. Leverage ({leverage:.2f}) exceeds max allowed ({max_leverage:.2f}).")
            return False
        return True
    return constraint

def minimum_volatility_constraint(min_volatility: float):
    """
    Ensures that the portfolio's volatility is above a specified minimum level.
    Volatility is calculated as the standard deviation of daily returns.
    """
    def constraint(data: dict, positions: np.ndarray, capital: float) -> bool:
        returns = data.get("Daily_Return", pd.Series(dtype=float))
        if returns.empty:
            print("Minimum volatility constraint failed. No returns available.")
            return False

        volatility = returns.std()
        print(f"Calculated volatility: {volatility:.6f}, Allowed min volatility: {min_volatility:.6f}")

        if volatility < min_volatility:
            print(f"Minimum volatility constraint failed. Volatility ({volatility:.6f}) is below minimum ({min_volatility:.6f}).")
            return False
        return True
    return constraint

def garch_constraint(max_volatility: float):
    """
    Ensures that the portfolio's forecasted volatility (using a GARCH model) does not exceed the specified maximum.
    GARCH models are used to forecast volatility based on historical data.
    """
    def constraint(data: dict, positions: np.ndarray, capital: float) -> bool:

        returns = data.get("Daily_Return", pd.Series(dtype=float))

        # Ensure sufficient observations
        if len(returns) < 10:
            print("GARCH constraint skipped due to insufficient data.")
            return True

        try:
            # Fit GARCH model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            forecast = fitted_model.forecast(horizon=1)

            # Extract forecast volatility
            forecast_volatility = np.sqrt(forecast.variance.values[-1, :][0])
            print(f"GARCH forecast volatility: {forecast_volatility:.6f}")

            # Check constraint
            if forecast_volatility > max_volatility:
                print(f"GARCH constraint failed. Forecast volatility ({forecast_volatility:.6f}) exceeds max ({max_volatility}).")
                return False
            return True

        except Exception as e:
            # Handle model fitting failures
            print(f"Error in GARCH model fitting: {e}")
            return False  # Conservatively fail the constraint
    
def portfolio_risk_constraint(max_risk: float):
    """
    Ensures that the portfolio's maximum drawdown does not exceed the specified risk level.
    Drawdown is calculated as the largest percentage drop from a peak to a trough in cumulative returns.
    """
    def constraint(data: dict, positions: np.ndarray, capital: float) -> bool:

        returns = data.get("Daily_Return", pd.Series(dtype=float))

        # Check if returns are empty
        if returns.empty:
            print("Portfolio risk constraint skipped due to empty returns.")
            return True

        try:
            # Calculate cumulative returns and drawdown
            cumulative_returns = (1 + returns).cumprod()
            drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
            max_drawdown = drawdown.max()

            # Log details
            print(f"Portfolio maximum drawdown: {max_drawdown:.6f}")
            print(f"Allowed maximum risk: {max_risk:.6f}")

            # Check if max drawdown exceeds the allowed risk
            if max_drawdown > max_risk:
                print(f"Portfolio risk constraint failed. Drawdown ({max_drawdown:.6f}) exceeds allowed risk ({max_risk:.6f}).")
                return False

            return True

        except Exception as e:
            # Handle calculation errors
            print(f"Error in portfolio risk constraint: {e}")
            return False  # Fail conservatively


def jump_risk_constraint(max_jump_risk: float):
    """
    Ensures that extreme price changes (measured as the 99th percentile of absolute daily returns) do not exceed the specified limit.
    """
    def constraint(data: dict, positions: np.ndarray, capital: float) -> bool:
        returns = data.get("Daily_Return", pd.Series(dtype=float))
        if returns.empty:
            print("Jump risk constraint skipped due to empty returns.")
            return True

        jump_risk = returns.abs().quantile(0.99)
        print(f"Calculated jump risk: {jump_risk:.6f}, Allowed max jump risk: {max_jump_risk:.6f}")

        if jump_risk > max_jump_risk:
            print(f"Jump risk constraint failed. Jump risk ({jump_risk:.6f}) exceeds max allowed ({max_jump_risk:.6f}).")
            return False
        return True
    return constraint


def concentration_constraint(max_concentration: float):
    """
    Ensures that no single position accounts for more than a specified percentage of the total portfolio value.
    Concentration is calculated as the ratio of the largest position value to the total portfolio value.
    """
    def constraint(data: dict, positions: np.ndarray, capital: float) -> bool:
        portfolio_value = data.get("portfolio_value", None)
        portfolio_value = portfolio_value.sum()  # Aggregate to a scalar

        if portfolio_value is None or portfolio_value == 0:
            print("Concentration constraint skipped due to missing or zero portfolio value.")
            return True

        # Calculate concentration for each position
        position_values = positions * data['close'].values
        max_position_value = max(position_values)
        concentration = max_position_value / portfolio_value
        print(f"Calculated Concentration: {concentration:.6f}, Allowed Max Concentration: {max_concentration:.6f}")

        if concentration > max_concentration:
            print(f"Concentration constraint failed. Concentration ({concentration:.6f}) exceeds max allowed ({max_concentration:.6f}).")
            return False
        return True
    return constraint

def liquidity_constraint(max_position_volume: float):
    """
    Ensures that the volume of a position does not exceed a specified percentage of the average daily trading volume.
    This constraint helps maintain liquidity and avoids market impact.
    """
    def constraint(data: dict, positions: np.ndarray, capital: float) -> bool:
        daily_volume = data.get("volume", pd.Series(dtype=float))
        if daily_volume.empty:
            print("Liquidity constraint skipped due to missing volume data.")
            return True

        # Check each position against the liquidity constraint
        for i, position in enumerate(positions):
            position_volume = position * data['close'].iloc[i]
            max_allowable_volume = max_position_volume * daily_volume.iloc[i]
            if position_volume > max_allowable_volume:
                print(f"Liquidity constraint failed. Position volume ({position_volume:.2f}) exceeds max allowable volume ({max_allowable_volume:.2f}).")
                return False
        return True
    return constraint

