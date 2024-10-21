import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def is_trading_day(day):
    """Check if the given day is a weekend."""
    return day.weekday() < 5  # Monday=0, Sunday=6

def get_last_trading_day():
    """Return the last trading day (skip weekends)."""
    today = datetime.now()
    # If today is Monday, get last Friday; otherwise, get the previous day
    last_trading_day = today - timedelta(days=3 if today.weekday() == 0 else 1)
    return last_trading_day.strftime('%Y-%m-%d')

def get_data(start_date='2010-01-01', end_date=None):
    """Retrieve OHLCV data for all tickers, updating the database if needed."""

    if end_date is None:
        # Use the last trading day as the end date if today is a weekend or Monday
        end_date = get_last_trading_day()

    # Ensure start_date is also in string format
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')

    db_name = 'sp500_stock_data.db'
    conn = sqlite3.connect(db_name)

    try:
        # Fetch all distinct tickers from the database
        tickers_query = "SELECT DISTINCT ticker FROM stock_data"
        tickers = pd.read_sql_query(tickers_query, conn)['ticker'].tolist()

        if not tickers:
            print("No tickers found in the database.")
            return None

        new_data_needed = False

        for ticker in tickers:
            # Get the latest available date for each ticker
            latest_date_query = "SELECT MAX(date) as latest_date FROM stock_data WHERE ticker = ?"
            latest_date_result = pd.read_sql_query(latest_date_query, conn, params=(ticker,))
            latest_date = latest_date_result.iloc[0]['latest_date']

            if latest_date:
                latest_date = pd.to_datetime(latest_date).strftime('%Y-%m-%d')
                if latest_date >= end_date:
                    print(f"No new data needed for {ticker}.")
                    continue  # Skip if we have the most recent data

                # Increment the latest date by 1 day
                latest_date = (datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                latest_date = start_date

            # Fetch new data if needed
            print(f"Fetching new data for {ticker} from {latest_date} to {end_date}...")
            new_data = yf.download(ticker, start=latest_date, end=end_date)

            if not new_data.empty:
                new_data.reset_index(inplace=True)
                new_data['ticker'] = ticker
                new_data = new_data[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
                new_data.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                new_data.to_sql('stock_data', conn, if_exists='append', index=False)
                new_data_needed = True

        if new_data_needed:
            print("Database updated with new data.")

        # Use SQLite's DATE() function for date comparisons
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

        # Convert 'date' column to datetime format
        all_data['date'] = pd.to_datetime(all_data['date'], errors='coerce')
        all_data['PercentChange'] = (all_data['close'] - all_data['open']) / all_data['open']

        # Set multi-index (ticker, date) and return as DataFrame
        all_data.set_index(['ticker', 'date'], inplace=True)

        return all_data

    except Exception as e:
        print(f"Error retrieving or updating stock data: {e}")
        return None

    finally:
        conn.close()
