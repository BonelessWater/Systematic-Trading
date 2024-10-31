import sqlite3
import pandas as pd
import yfinance as yf

# Create or connect to the SQLite3 database
db_name = 'sp500_stock_data.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Define the table schema
cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adjusted_close REAL,
        volume INTEGER,
        PRIMARY KEY (ticker, date)
    )
''')

# Fetch the list of S&P 500 companies
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

# Define the period for which we need data
start_date = '2015-01-01'
end_date = '2024-10-28'

# Loop through each S&P 500 stock ticker to fetch and store data
for ticker in sp500_tickers:
    try:
        print(f"Processing {ticker}...")

        # Check if data already exists in the database for the given ticker and date range
        cursor.execute(
            "SELECT date FROM stock_data WHERE ticker = ? AND date >= ? AND date <= ?",
            (ticker, start_date, end_date)
        )
        existing_dates = set(row[0] for row in cursor.fetchall())

        # Fetch historical data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"No data for {ticker}, skipping.")
            continue

        # Handle potential multi-index issue by flattening the columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Verify the structure and print the columns for debugging
        print(f"Columns for {ticker}: {data.columns.tolist()}")

        # Define required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Error processing {ticker}: Missing columns {missing_columns}")
            continue

        # Drop rows with missing values
        data = data.dropna(subset=required_columns)

        if data.empty:
            print(f"No valid data for {ticker} after dropping missing values, skipping.")
            continue

        # Insert new data into the database
        for index, row in data.iterrows():
            date_str = index.strftime('%Y-%m-%d')
            if date_str not in existing_dates:
                cursor.execute(
                    '''
                    INSERT INTO stock_data (ticker, date, open, high, low, close, adjusted_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        ticker,
                        date_str,
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        float(row['Adj Close']),
                        int(row['Volume'])
                    )
                )

        conn.commit()  # Commit changes to the database

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Close the database connection
conn.close()
print("Data logging completed.")
