import numpy as np
import datetime
import logging
from typing import Callable

from risk_management.utility_functions._logging import LogMessage, LogSubType, LogType

def minimum_volatility(max_forecast_ratio : float, IDM : float, tau : float, maximum_leverage : float, instrument_weight : float | np.ndarray, annualized_volatility : float | np.ndarray) -> bool:
    """
    Returns True if the returns for a given instrument meets a minimum level of volatility; else, False
    (works for both single instruments and arrays)

    Parameters:
    ---
        max_forecast_ratio : float
            the max forecast ratio (max forecast / average forecast) ... often 20 / 10 -> 2
        IDM : float
            instrument diversification multiplier
        tau : float
            the target risk for the portfolio
        maximum_leverage : float
            the max acceptable leverage for a given instrument
        instrument_weight : float | np.ndarray
            the weight of the instrument in the portfolio (capital allocated to the instrument / total capital)
            ... often 1/N
        annualized_volatility : float | np.ndarray
            standard deviation of returns for the instrument, in same terms as tau e.g. annualized
    """
    return annualized_volatility >= (max_forecast_ratio * IDM * instrument_weight * tau) / maximum_leverage

def portfolio_multiplier(
        max_portfolio_leverage : float, 
        max_correlation_risk : float, 
        max_portfolio_volatility : float,
        max_portfolio_jump_risk : float) -> Callable:

    def max_leverage(positions_weighted : np.ndarray) -> float:
        """
        Parameters:
        ---
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
        """
        leverage = np.sum(np.abs(positions_weighted))
        if leverage == 0:
            return np.float64(1)
        return min(max_portfolio_leverage / leverage, np.float64(1))

    def correlation_risk(positions_weighted : np.ndarray, annualized_volatility : np.ndarray) -> float:
        """
        Parameters:
        ---
            positions_weighted : np.ndarray (dollars allocated to each instrument)
                the notional exposure * # positions / capital
                Same as dynamic optimization
            annualized_volatility : np.ndarray
                standard deviation of returns for the instrument, in same terms as tau e.g. annualized
        """
        correlation_risk = np.sum(np.abs(positions_weighted) * annualized_volatility)
        if correlation_risk == 0:
            return np.float64(1)
        return min(max_correlation_risk / correlation_risk, np.float64(1))
    
    def portfolio_risk(positions_weighted : np.ndarray, covariance_matrix : np.ndarray) -> float:
        """
        Parameters:
        ---
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
            covariance_matrix : np.ndarray
                the covariances between the instrument returns
        """
        portfolio_volatility = np.sqrt(positions_weighted @ covariance_matrix @ positions_weighted.T)
        if portfolio_volatility == 0:
            return np.float64(1)
        return min(max_portfolio_volatility / portfolio_volatility, np.float64(1))

    def jump_risk_multiplier(positions_weighted : np.ndarray, jump_covariance_matrix : np.ndarray) -> float:
        """
        Parameters:
        ---
            maximum_portfolio_jump_risk : float
                the max acceptable jump risk for the portfolio
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
            jumps : np.ndarray
                the jumps in the instrument returns
        """
        jump_risk = np.sqrt(positions_weighted @ jump_covariance_matrix @ positions_weighted.T)
        if jump_risk == 0:
            return np.float64(1)
        return min(max_portfolio_jump_risk / jump_risk, np.float64(1))

    def fn(
            positions_weighted : np.ndarray,
            covariance_matrix : np.ndarray,
            jump_covariance_matrix : np.ndarray,
            date : datetime.datetime) -> float:

        annualized_volatility = np.diag(covariance_matrix) * 256 ** 0.5

        scalars = {
            LogSubType.LEVERAGE_MULTIPLIER : max_leverage(positions_weighted), 
            LogSubType.CORRELATION_MULTIPLIER : correlation_risk(positions_weighted, annualized_volatility),
            LogSubType.VOLATILITY_MULTIPLIER : portfolio_risk(positions_weighted, covariance_matrix),
            LogSubType.JUMP_MULTIPLIER : jump_risk_multiplier(positions_weighted, jump_covariance_matrix)}

        portfolio_scalar = np.float64(1)
        for key, value in scalars.items():
            if value < 1:
                portfolio_scalar = value
                logging.warning(LogMessage(date, LogType.PORTFOLIO_MULTIPLIER, key, None, value))

        return portfolio_scalar

    return fn

def position_limit(
        max_leverage_ratio : int,
        minimum_volume : int,
        max_forecast_ratio : float,
        max_forecast_buffer : float,
        IDM : float,
        tau : float) -> Callable:

    def max_leverage(capital : float, notional_exposure_per_contract : np.ndarray) -> np.ndarray:
        """
        Parameters:
        ---
            maximum_leverage : float
                the max acceptable leverage for a given instrument
            capital : float
                the total capital allocated to the portfolio
            notional_exposure_per_contract : np.ndarray
                the notional exposure per contract for the instrument
        """
        return max_leverage_ratio * capital / notional_exposure_per_contract

    def max_forecast(capital : float, notional_exposure_per_contract : np.ndarray, instrument_weight : np.ndarray, annualized_volatility : np.ndarray) -> np.ndarray:
        """
        Parameters:
        ---
            maximum_forecast_ratio : float
                the max forecast ratio (max forecast / average forecast) ... often 20 / 10 -> 2
            capital : float
                the total capital allocated to the portfolio
            IDM : float
                instrument diversification multiplier
            tau : float
                the target risk for the portfolio
            instrument_weight : float | np.ndarray
                the weight of the instrument in the portfolio (capital allocated to the instrument / total capital)
                ... often 1/N
            notional_exposure_per_contract : float | np.ndarray
                the notional exposure per contract for the instrument
            annualized_volatility : float | np.ndarray
                standard deviation of returns for the instrument, in same terms as tau e.g. annualized
        """
        return (1 + max_forecast_buffer) * max_forecast_ratio * capital * IDM * instrument_weight * tau / notional_exposure_per_contract / annualized_volatility

    def min_volume(volume : np.ndarray) -> np.ndarray:
        """
        Parameters:
        ---
            volume : float | np.ndarray
                the volume for the instrument
            minimum_volume : float
                minimum volume requirement for any instrument
        """
        volume_mask = np.where(volume < minimum_volume, 0, 1)

        return volume_mask

    def fn(
            capital : float,
            positions : np.ndarray,
            notional_exposure_per_contract : np.ndarray,
            instrument_weight : np.ndarray,
            covariance_matrix : np.ndarray,
            volume : np.ndarray,
            additional_data : tuple[list[str], datetime.datetime]):

        annualized_volatility = np.diag(covariance_matrix) * 256 ** 0.5

        positions_at_maximum_leverage = abs(max_leverage(capital, notional_exposure_per_contract))
        positions_at_maximum_forecast = abs(max_forecast(capital, notional_exposure_per_contract, instrument_weight, annualized_volatility))
        volume_mask = min_volume(volume)

        max_positions =  volume_mask * np.minimum(positions_at_maximum_leverage, positions_at_maximum_forecast)

        for idx, _ in enumerate(volume_mask):
            if volume_mask[idx] == 0:
                logging.warning(
                    LogMessage(
                        additional_data[1],
                        LogType.POSITION_LIMIT,
                        LogSubType.MINIMUM_VOLUME,
                        additional_data[0][idx],
                        np.float64(0)))

        for position_at_maximum_leverage, position in zip(positions_at_maximum_leverage, positions):
            if abs(position) > position_at_maximum_leverage:
                logging.warning(
                    LogMessage(
                        additional_data[1],
                        LogType.POSITION_LIMIT,
                        LogSubType.MAX_LEVERAGE,
                        additional_data[0][np.where(positions == position)[0][0]],
                        position_at_maximum_leverage))
         
        for position_at_maximum_forecast, position in zip(positions_at_maximum_forecast, positions):
            if abs(position) > position_at_maximum_forecast:
                logging.warning(
                    LogMessage(
                        additional_data[1],
                        LogType.POSITION_LIMIT,
                        LogSubType.MAX_FORECAST,
                        additional_data[0][np.where(positions == position)[0][0]],
                        position_at_maximum_forecast))
     
        sign_map = np.sign(positions)
        minimum_position = np.minimum(abs(positions), max_positions) * sign_map

        return minimum_position

    return fn
