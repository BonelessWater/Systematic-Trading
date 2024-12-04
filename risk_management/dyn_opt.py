import os 

# print(os.getcwd())
# print(os.listdir('risk_limits'))
# print(os.listdir('dyn_opt'))
# import sys
# print(sys.path)
# import sys
# sys.path.append('/app/risk_limits')


import pandas as pd
import numpy as np
import logging
import datetime
import pickle
import sys
from functools import reduce
from risk_management.risk_limits.risk_limit import portfolio_risk, position_risk
from risk_management.utility_functions._logging import CsvFormatter

from risk_management.risk_measures.risk_measures.risk_functions import daily_variance_to_annualized_volatility


logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler('log.csv', mode='w'),
        logging.StreamHandler()])

logger = logging.getLogger(__name__)
logging.root.handlers[0].setFormatter(CsvFormatter())

def get_notional_exposure_per_contract(unadj_prices : pd.DataFrame, multipliers : pd.DataFrame) -> pd.DataFrame:
    notional_exposure_per_contract = unadj_prices.apply(lambda col: col * multipliers.loc['Multiplier', col.name])
    return notional_exposure_per_contract.abs()

def get_weight_per_contract(notional_exposure_per_contract : pd.DataFrame, capital : float) -> pd.DataFrame:
    return notional_exposure_per_contract / capital

def get_cost_penalty(x_weighted : np.ndarray, y_weighted : np.ndarray, weighted_cost_per_contract : np.ndarray, cost_penalty_scalar : int) -> float:
    """Finds the trading cost to go from x to y, given the weighted cost per contract and the cost penalty scalar"""

    #* Should never activate but just in case
    # x_weighted = np.nan_to_num(np.asarray(x_weighted, dtype=np.float64))
    # y_weighted = np.nan_to_num(np.asarray(y_weighted, dtype=np.float64))
    # weighted_cost_per_contract = np.nan_to_num(np.asarray(weighted_cost_per_contract, dtype=np.float64))

    trading_cost = np.abs(x_weighted - y_weighted) * weighted_cost_per_contract

    return np.sum(trading_cost) * cost_penalty_scalar

def get_portfolio_tracking_error_standard_deviation(x_weighted : np.ndarray, y_weighted : np.ndarray, covariance_matrix : np.ndarray, cost_penalty : float = 0.0) -> float:
    if np.isnan(x_weighted).any() or np.isnan(y_weighted).any() or np.isnan(covariance_matrix).any():
        raise ValueError("Input contains NaN values")
    
    tracking_errors = x_weighted - y_weighted

    dot_product = np.dot(np.dot(tracking_errors, covariance_matrix), tracking_errors)

    #* deal with negative radicand (really, REALLY shouldn't happen)
    #? maybe its a good weight set but for now, it's probably safer this way
    if dot_product < 0:
        return 1.0
    
    return np.sqrt(dot_product) + cost_penalty

def covariance_row_to_matrix(row : np.ndarray) -> np.ndarray:
    num_instruments = int(np.sqrt(2 * len(row)))
    matrix = np.zeros((num_instruments, num_instruments))

    idx = 0
    for i in range(num_instruments):
        for j in range(i, num_instruments):
            matrix[i, j] = matrix[j, i] = row[idx]
            idx += 1

    return matrix

def round_multiple(x : np.ndarray, multiple : np.ndarray) -> np.ndarray:
    return np.round(x / multiple) * multiple

def buffer_weights(optimized : np.ndarray, held : np.ndarray, weights : np.ndarray, covariance_matrix : np.ndarray, tau : float, asymmetric_risk_buffer : float):
    tracking_error = get_portfolio_tracking_error_standard_deviation(optimized, held, covariance_matrix)

    tracking_error_buffer = tau * asymmetric_risk_buffer

    # If the tracking error is less than the buffer, we don't need to do anything
    if tracking_error < tracking_error_buffer:
        return held
    
    adjustment_factor = max((tracking_error - tracking_error_buffer) / tracking_error, 0.0)

    required_trades = (optimized - held) * adjustment_factor

    return round_multiple(held + required_trades, weights)

# Might be worth framing this similar to scipy.minimize function in terms of argument names (or quite frankly, maybe just use scipy.minimize)
def greedy_algorithm(ideal : np.ndarray, x0 : np.ndarray, weighted_costs_per_contract : np.ndarray, held : np.ndarray, weights_per_contract : np.ndarray, covariance_matrix : np.ndarray, cost_penalty_scalar : int) -> np.ndarray:
    if ideal.ndim != 1 or ideal.shape != x0.shape != held.shape != weights_per_contract.shape != weighted_costs_per_contract.shape:
        raise ValueError("Input shapes do not match")
    if covariance_matrix.ndim != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1] or len(ideal) != covariance_matrix.shape[0]:
        raise ValueError("Invalid covariance matrix (should be [N x N])")
    
    proposed_solution = x0.copy()
    cost_penalty = get_cost_penalty(held, proposed_solution, weighted_costs_per_contract, cost_penalty_scalar)
    tracking_error = get_portfolio_tracking_error_standard_deviation(ideal, proposed_solution, covariance_matrix, cost_penalty)
    best_tracking_error = tracking_error
    iteration_limit = 1000
    iteration = 0

    while iteration <= iteration_limit:
        previous_solution = proposed_solution.copy()
        best_IDX = None

        for idx in range(len(proposed_solution)):
            temp_solution = previous_solution.copy()

            if temp_solution[idx] < ideal[idx]:
                temp_solution[idx] += weights_per_contract[idx]
            else:
                temp_solution[idx] -= weights_per_contract[idx]

            cost_penalty = get_cost_penalty(held, temp_solution, weighted_costs_per_contract, cost_penalty_scalar)
            tracking_error = get_portfolio_tracking_error_standard_deviation(ideal, temp_solution, covariance_matrix, cost_penalty)

            if tracking_error <= best_tracking_error:
                best_tracking_error = tracking_error
                best_IDX = idx

        if best_IDX is None:
            break

        if proposed_solution[best_IDX] <= ideal[best_IDX]:
            proposed_solution[best_IDX] += weights_per_contract[best_IDX]
        else:
            proposed_solution[best_IDX] -= weights_per_contract[best_IDX]
        
        iteration += 1

    return proposed_solution

def clean_data(*args):
    dfs = [df.set_index(pd.to_datetime(df.index)).dropna() for df in args]
    intersection_index = reduce(lambda x, y: x.intersection(y), (df.index for df in dfs))
    dfs = [df.loc[intersection_index] for df in dfs]

    return dfs

def single_day_optimized_positions(
        covariances_one_day : np.ndarray,
        jump_covariances_one_day : np.ndarray,
        held_positions_one_day : np.ndarray,
        ideal_positions_one_day : np.ndarray,
        weight_per_contract_one_day : np.ndarray,
        costs_per_contract_one_day : np.ndarray,
        notional_exposure_per_contract_one_day : np.ndarray,
        open_interest_one_day : np.ndarray,
        instrument_weight_one_day : np.ndarray,
        tau : float,
        capital : float,
        IDM : float,
        maximum_forecast_ratio : float,
        maximum_position_leverage : float,
        max_acceptable_pct_of_open_interest : float,
        max_forecast_buffer : float,
        maximum_portfolio_leverage : float,
        maximum_correlation_risk : float,
        maximum_portfolio_risk : float,
        maximum_jump_risk : float,
        asymmetric_risk_buffer : float,
        cost_penalty_scalar : int,
        additional_data : tuple[list[str], list[datetime.datetime]]) -> np.ndarray:
    covariance_matrix_one_day : np.ndarray = covariance_row_to_matrix(covariances_one_day)
    jump_covariance_matrix_one_day : np.ndarray = covariance_row_to_matrix(jump_covariances_one_day)

    ideal_positions_weighted = ideal_positions_one_day * weight_per_contract_one_day
    held_positions_weighted = held_positions_one_day * weight_per_contract_one_day
    costs_per_contract_weighted = costs_per_contract_one_day / capital / weight_per_contract_one_day

    x0 : np.ndarray = held_positions_weighted

    optimized_weights_one_day = greedy_algorithm(ideal_positions_weighted, x0, costs_per_contract_weighted, held_positions_weighted, weight_per_contract_one_day, covariance_matrix_one_day, cost_penalty_scalar)

    buffered_weights = buffer_weights(
        optimized_weights_one_day, held_positions_weighted, weight_per_contract_one_day, 
        covariance_matrix_one_day, tau, asymmetric_risk_buffer)

    optimized_positions_one_day = buffered_weights / weight_per_contract_one_day

    annualized_volatilities = daily_variance_to_annualized_volatility(np.diag(covariance_matrix_one_day))

    risk_limited_positions = position_risk.position_limit_aggregator(
        maximum_position_leverage, capital, IDM, tau, maximum_forecast_ratio, 
        max_acceptable_pct_of_open_interest, max_forecast_buffer, optimized_positions_one_day, 
        notional_exposure_per_contract_one_day, annualized_volatilities, instrument_weight_one_day, open_interest_one_day, additional_data)

    risk_limited_positions_weighted = risk_limited_positions * weight_per_contract_one_day

    portfolio_risk_limited_positions = portfolio_risk.portfolio_risk_aggregator(
        risk_limited_positions, risk_limited_positions_weighted, covariance_matrix_one_day, 
        jump_covariance_matrix_one_day, maximum_portfolio_leverage, maximum_correlation_risk, 
        maximum_portfolio_risk, maximum_jump_risk, date=additional_data[1])

    return round_multiple(portfolio_risk_limited_positions, 1)

def iterator(
        covariances : pd.DataFrame,
        jump_covariances : pd.DataFrame,
        ideal_positions : pd.DataFrame, 
        weight_per_contract : pd.DataFrame, 
        costs_per_contract : pd.DataFrame,
        notional_exposure_per_contract : pd.DataFrame,
        open_interest : pd.DataFrame,
        instrument_weight : pd.DataFrame,
        tau : float,
        capital : float,
        IDM : float,
        maximum_forecast_ratio : float,
        maximum_position_leverage : float,
        max_acceptable_pct_of_open_interest : float,
        max_forecast_buffer : float,
        maximum_portfolio_leverage : float,
        maximum_correlation_risk : float, 
        maximum_portfolio_risk : float,
        maximum_jump_risk : float,
        asymmetric_risk_buffer : float,
        cost_penalty_scalar : int = 10) -> pd.DataFrame:
    #@ Data cleaning
    ideal_positions, weight_per_contract, costs_per_contract, covariances = clean_data(ideal_positions, weight_per_contract, costs_per_contract, covariances)

    # Make sure they all have the same columns, and order !!
    intersection_columns = ideal_positions.columns.intersection(weight_per_contract.columns).intersection(costs_per_contract.columns)
    ideal_positions = ideal_positions[intersection_columns]
    weight_per_contract = weight_per_contract[intersection_columns]
    costs_per_contract = costs_per_contract[intersection_columns]

    optimized_positions = pd.DataFrame(index=ideal_positions.index, columns=ideal_positions.columns)
    optimized_positions = optimized_positions.astype(np.float64)

    vectorized_ideal_positions = ideal_positions.values
    vectorized_costs_per_contract = costs_per_contract.values
    vectorized_weight_per_contract = weight_per_contract.values
    vectorized_notional_exposure_per_contract = notional_exposure_per_contract.values
    vectorized_open_interest = open_interest.values
    vectorized_covariances = covariances.values
    vectorized_jump_covariances = jump_covariances.values
    vectorized_instrument_weight = instrument_weight.values

    for n, date in enumerate(ideal_positions.index):
        held_positions = np.zeros(len(ideal_positions.columns))

        if n != 0:
            current_date_IDX = ideal_positions.index.get_loc(date)
            held_positions = optimized_positions.iloc[current_date_IDX - 1].values

        optimized_positions.loc[date] = single_day_optimized_positions(
            vectorized_covariances[n], vectorized_jump_covariances[n], held_positions, vectorized_ideal_positions[n], 
            vectorized_weight_per_contract[n], vectorized_costs_per_contract[n], vectorized_notional_exposure_per_contract[n], 
            vectorized_open_interest[n], vectorized_instrument_weight[n], tau, capital, IDM, maximum_forecast_ratio, 
            maximum_position_leverage, max_acceptable_pct_of_open_interest, max_forecast_buffer, maximum_portfolio_leverage, 
            maximum_correlation_risk, maximum_portfolio_risk, maximum_jump_risk, asymmetric_risk_buffer, cost_penalty_scalar, (ideal_positions.columns, date))

    return optimized_positions

def aggregator(
    capital : float,
    fixed_cost_per_contract : float,
    tau : float,
    asymmetric_risk_buffer : float,
    unadj_prices : pd.DataFrame,
    multipliers : pd.DataFrame,
    ideal_positions : pd.DataFrame,
    covariances : pd.DataFrame,
    jump_covariances : pd.DataFrame,
    open_interest : pd.DataFrame,
    instrument_weight : pd.DataFrame,
    IDM : float,
    maximum_forecast_ratio : float,
    max_acceptable_pct_of_open_interest : float,
    max_forecast_buffer : float,
    maximum_position_leverage : float,
    maximum_portfolio_leverage : float,
    maximum_correlation_risk : float,
    maximum_portfolio_risk : float,
    maximum_jump_risk : float,
    cost_penalty_scalar : int)-> pd.DataFrame:

    unadj_prices, ideal_positions, covariances, jump_covariances, open_interest, instrument_weight = clean_data(unadj_prices, ideal_positions, covariances, jump_covariances, open_interest, instrument_weight)

    multipliers = multipliers.sort_index(axis=1)

    notional_exposure_per_contract = get_notional_exposure_per_contract(unadj_prices, multipliers)
    weight_per_contract = get_weight_per_contract(notional_exposure_per_contract, capital)

    costs_per_contract = pd.DataFrame(index=ideal_positions.index, columns=ideal_positions.columns).fillna(fixed_cost_per_contract)

    return iterator(
        covariances, jump_covariances, ideal_positions, weight_per_contract,
        costs_per_contract, notional_exposure_per_contract, open_interest, 
        instrument_weight, tau, capital, IDM, maximum_forecast_ratio, maximum_position_leverage, 
        max_acceptable_pct_of_open_interest, max_forecast_buffer, maximum_portfolio_leverage, 
        maximum_correlation_risk, maximum_portfolio_risk, maximum_jump_risk, asymmetric_risk_buffer, cost_penalty_scalar)
