"""
High-Dividend Low-Volatility Strategy: Data Utilities
====================================================

This module handles loading and preparation of data for the backtesting framework.
"""

import pandas as pd
import numpy as np
import time
import yfinance as yf
import pandas_datareader.data as web


def load_sp500_constituents(csv_file_path, target_date):
    """
    Load S&P 500 constituents from a CSV file as of a specified date.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing historical S&P 500 constituents
    target_date : str
        Date for which to get the S&P 500 constituents (before backtest start)
        
    Returns:
    --------
    list
        List of S&P 500 ticker symbols as of the target date
    """
    print(f"Loading S&P 500 constituents as of {target_date}...")
    
    # Convert target_date to datetime
    target_date = pd.to_datetime(target_date)
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded CSV file with {len(df)} rows")
        
        # Convert date column to datetime if it's not already
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            print(f"Warning: 'date' column not found in CSV. Available columns: {df.columns.tolist()}")
            return None
        
        # Find the closest date in the dataframe that's less than or equal to the target date
        closest_date = df[df['date'] <= target_date]['date'].max()
        
        if pd.isna(closest_date):
            print(f"No data found before or on {target_date}")
            return None
        
        print(f"Using constituents from {closest_date.strftime('%Y-%m-%d')} (closest to target {target_date.strftime('%Y-%m-%d')})")
        
        # Get the row for the closest date
        constituents_row = df[df['date'] == closest_date].iloc[0]
        
        # Extract the tickers from the 'tickers' column
        if 'tickers' in constituents_row:
            # Split the comma-separated string of tickers into a list
            sp500_tickers = constituents_row['tickers'].split(',')
            sp500_tickers = [ticker.strip() for ticker in sp500_tickers]  # Remove any whitespace
            print(f"Found {len(sp500_tickers)} constituents")
            return sp500_tickers
        else:
            print(f"Warning: 'tickers' column not found. Available columns: {constituents_row.index.tolist()}")
            return None
            
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def fetch_stock_data(tickers, start_date, end_date, batch_size=50):
    """
    Fetch historical price and dividend data for a list of tickers.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols to fetch data for
    start_date : str
        Start date for data fetching
    end_date : str
        End date for data fetching
    batch_size : int, optional
        Number of tickers to process in each batch (to avoid API rate limits)
        
    Returns:
    --------
    tuple
        (prices_dict, dividends_dict, valid_tickers, invalid_tickers)
        where prices_dict and dividends_dict are dictionaries mapping tickers to price and dividend Series
    """
    print(f"Fetching data for {len(tickers)} tickers... (this may take a while)")
    start_time = time.time()
    
    # Fetch data
    prices = {}
    dividends = {}
    valid_tickers = []
    invalid_tickers = []
    
    # Add buffer to start date for TTM calculations
    extended_start = pd.Timestamp(start_date) - pd.DateOffset(years=1, days=10)
    extended_start_str = extended_start.strftime('%Y-%m-%d')
    
    # Process in smaller batches to avoid rate limits
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)")
        
        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=extended_start_str, end=end_date)
                
                if not hist.empty and 'Close' in hist.columns:
                    # Convert timezone-aware DatetimeIndex to timezone-naive
                    hist.index = hist.index.tz_localize(None)
                    
                    # Forward fill prices to handle missing days
                    prices[ticker] = hist["Close"].resample('B').ffill()
                    
                    # Resample dividends to daily with zero-filling
                    if "Dividends" in hist.columns:
                        full_idx = pd.date_range(start=hist.index.min(), end=hist.index.max(), freq='B')
                        div_series = hist["Dividends"].reindex(full_idx, fill_value=0)
                        dividends[ticker] = div_series
                    else:
                        full_idx = pd.date_range(start=hist.index.min(), end=hist.index.max(), freq='B')
                        dividends[ticker] = pd.Series(0, index=full_idx)
                    
                    # Add to valid tickers
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
                    print(f"Warning: No valid data found for {ticker}")
            except Exception as e:
                invalid_tickers.append(ticker)
                print(f"Error fetching data for {ticker}: {str(e)[:100]}")  # Truncate long error messages
        
        # Add a small delay between batches to avoid rate limits
        if batch_num < total_batches:
            time.sleep(1)
    
    data_fetch_time = time.time() - start_time
    print(f"\nData fetching completed in {data_fetch_time:.1f} seconds")
    print(f"Successfully fetched data for {len(valid_tickers)} tickers")
    print(f"Unable to fetch data for {len(invalid_tickers)} tickers")
    
    return prices, dividends, valid_tickers, invalid_tickers


def fetch_risk_free_rate(start_date, end_date):
    """
    Fetch risk-free rate data from FRED (3-Month Treasury Bill).
    
    Parameters:
    -----------
    start_date : str
        Start date for data fetching
    end_date : str
        End date for data fetching
        
    Returns:
    --------
    pandas.Series
        Daily risk-free rate (as a decimal, e.g., 0.0001 for 0.01% daily)
    """
    print("Fetching risk-free rate data...")
    try:
        extended_start = pd.Timestamp(start_date) - pd.DateOffset(days=30)
        rf = web.DataReader('TB3MS', 'fred', extended_start, end_date)
        risk_free = rf['TB3MS'] / 100 / 252  # Convert to daily rate
        risk_free = risk_free.ffill()
        print("Risk-free rate data fetched successfully")
        return risk_free
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        # Create a default risk-free rate series
        idx = pd.date_range(start=start_date, end=end_date, freq='B')
        risk_free = pd.Series(0.02/252, index=idx)  # Default 2% annual rate
        print("Using default risk-free rate of 2% annually")
        return risk_free
