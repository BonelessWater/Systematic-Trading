from trading_system import TradingSystem
from strategies.strategy1 import Strategy1
from strategies.strategy2 import Strategy2
from data import get_data, add_tickers_and_data
from init_db import init_db
import pandas as pd
import quantstats as qs

# input wrapper
def algo_lens():
    #! Run if database is not initialized
    # init_db()

    #! Helper function to add new tickers
    # add_tickers_and_data(tickers)

    # Get data, end_date defaults to today's date
    print('Fetching data')
    data : pd.DataFrame = get_data(start_date='2020-1-1', fetch=False)
    print('Data fetched')

    risk_target = 0.30 # Risky
    capital = 100000
    
    trading_system = TradingSystem(
        strategies=[
            #(Proportion of capital for strategy, Strategy Class)
            # (1.0, Strategy1(data=data, risk_target=risk_target, capital=capital, num_stocks=5)),
            (1.0, Strategy2(data=data, risk_target=risk_target, capital=capital, num_stocks=5)),
        ]
    )   

    # Run the backtest for the trading system
    print('Running backtest...')
    results = trading_system.backtest()

    # Extract results for the specified strategy
    strategy_results = results.get('Strategy2', {})

    # Convert to a DataFrame if it isn't already
    if not isinstance(strategy_results, pd.DataFrame):
        strategy_results = pd.DataFrame(strategy_results)

    # Ensure the 'date' column exists and set it as the index
    if 'date' in strategy_results.columns:
        strategy_results['date'] = pd.to_datetime(strategy_results['date'])  # Parse as datetime
        strategy_results = strategy_results.set_index('date')

    # Drop unwanted columns (e.g., Realized_Capital)
    columns_to_keep = ['Ideal_Capital']  # Specify the column to retain
    strategy_results = strategy_results[columns_to_keep]

    # Rename index and column for clarity
    strategy_results.index.name = 'Date'
    strategy_results.columns = ['Value']  # Rename column as needed

    # Ensure missing dates have NaN values if a complete date range is required
    strategy_results = strategy_results.asfreq('B')  # Fill missing business days

    # Convert to percent change
    strategy_results['Percent_Change'] = strategy_results['Value'].pct_change()

    # Drop the original 'Value' column if only percent change is needed
    strategy_results = strategy_results[['Percent_Change']]

    print(strategy_results)
    return strategy_results


