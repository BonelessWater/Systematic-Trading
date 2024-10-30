import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def is_trading_day(day):
    """Check if the given day is a weekday (not a weekend)."""
    return day.weekday() < 5

def get_last_trading_day():
    """Return the last trading day (skip weekends)."""
    today = datetime.now()
    if is_trading_day(today):
        return today.strftime('%Y-%m-%d')
    else:
        # If today is a weekend, return the previous Friday
        last_trading_day = today - timedelta(days=today.weekday() - 4)
        return last_trading_day.strftime('%Y-%m-%d')

def get_data(fetch=False, start_date='2010-01-01'):
    """Retrieve OHLCV data with an option to fetch the latest data."""
    end_date = get_last_trading_day()
    db_name = 'sp500_stock_data.db'
    conn = sqlite3.connect(db_name)

    try:
        # Check the latest date available in the database
        latest_date_query = "SELECT MAX(date) as latest_date FROM stock_data"
        latest_date_result = pd.read_sql_query(latest_date_query, conn)
        latest_date_in_db = pd.to_datetime(latest_date_result.iloc[0]['latest_date'])

        # If fetch is False, just return the available data
        if not fetch:
            print("Fetching skipped. Returning available data from the database...")
            return fetch_data_from_db(conn, start_date, end_date)

        # If data is up-to-date, return it without fetching new data
        if latest_date_in_db and latest_date_in_db.strftime('%Y-%m-%d') >= end_date:
            print(f"Data is up to date for {end_date}.")
            return fetch_data_from_db(conn, start_date, end_date)

        # Otherwise, fetch new data and update the database
        print(f"Fetching data from {latest_date_in_db + timedelta(days=1)} to {end_date}...")
        return update_and_fetch_data(conn, latest_date_in_db + timedelta(days=1), end_date)

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    finally:
        conn.close()

def update_and_fetch_data(conn, start_date, end_date):
    """Fetch and update the database with new data."""
    tickers_query = "SELECT DISTINCT ticker FROM stock_data"
    tickers = pd.read_sql_query(tickers_query, conn)['ticker'].tolist()

    for ticker in tickers:
        try:
            print(f"Fetching new data for {ticker} from {start_date} to {end_date}...")
            new_data = yf.download(ticker, start=start_date, end=end_date)

            if new_data.empty:
                print(f"No new data found for {ticker}. Skipping...")
                continue  # Skip if no data is available

            new_data.reset_index(inplace=True)
            new_data['ticker'] = ticker
            new_data = new_data[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            new_data.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            new_data.to_sql('stock_data', conn, if_exists='append', index=False)

        except yf.YFPricesMissingError:
            print(f"{ticker} may be delisted or no price data available.")
            continue  # Skip if the ticker is delisted

    return fetch_data_from_db(conn, start_date, end_date)

def fetch_data_from_db(conn, start_date, end_date):
    """Fetch the required data from the database."""
    query = """
    SELECT ticker, DATE(date) as date, open, high, low, close, volume
    FROM stock_data
    WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)
    ORDER BY ticker, date
    """
    all_data = pd.read_sql_query(query, conn, params=(start_date, end_date))

    if all_data.empty:
        print("No data available in the specified range.")
        return None

    # Ensure 'date' is in datetime format
    all_data['date'] = pd.to_datetime(all_data['date'], errors='coerce')

    # Add 'PercentChange' column
    all_data['PercentChange'] = (all_data['close'] - all_data['open']) / all_data['open']

    # Keep 'ticker' as a column, not just part of the index
    all_data = all_data[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'PercentChange']]
    print(f"Fetched {len(all_data)} rows of data.")
    
    return all_data
