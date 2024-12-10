from trading_system import TradingSystem
from strategies.strategy1 import Strategy1
from strategies.strategy2 import Strategy2
from data import get_data, add_tickers_and_data
from init_db import init_db
from datetime import datetime
import pandas as pd
from typing import Callable, Any, List, Dict
import numpy as np

if __name__ == "__main__":
    #! Run if database is not initialized
    # init_db()

    #! Helper function to add new tickers
    # add_tickers_and_data(tickers)


    # Get data, end_date defaults to today's date
    print('Fetching data')
    data : pd.DataFrame = get_data(start_date='2024-1-1', fetch=False)
    print('Data fetched')

    risk_target = 0.30 # Risky
    capital = 100000 # One Thousand USD

    trading_system = TradingSystem(
        strategies=[
            #(Proportion of capital for strategy, Strategy Class)
            # (1.0, Strategy1(data=data, risk_target=risk_target, capital=capital, num_stocks=5)),
            (1.0, Strategy2(data=data, risk_target=risk_target, capital=capital, num_stocks=5)),
        ]
    )   

    trading_system.backtest()
    #trading_system.graph()
    trading_system.metrics()
    trading_system.plot_pnl(save_path='pnl_plot.png', log_scale=True)



