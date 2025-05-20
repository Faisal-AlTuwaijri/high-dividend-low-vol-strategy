"""
High-Dividend Low-Volatility Strategy: Stock Selection
=====================================================

This module contains functions for selecting stocks based on dividend
yield and volatility criteria.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def calculate_dividend_yields(prices, dividends, date, lookback_days=365):
    """
    Calculate trailing twelve-month dividend yields for a set of stocks.
    
    Parameters:
    -----------
    prices : dict
        Dictionary mapping tickers to price Series
    dividends : dict
        Dictionary mapping tickers to dividend Series
    date : datetime
        Date as of which to calculate yields
    lookback_days : int, optional
        Number of days to look back for dividend calculation
        
    Returns:
    --------
    dict
        Dictionary mapping tickers to their dividend yields
    """
    yields = {}
    lookback_start = date - timedelta(days=lookback_days)
    
    for ticker in prices.keys():
        # Make sure we have data for this ticker
        if ticker not in prices or ticker not in dividends:
            continue
            
        # Filter price series to the lookback period
        price_mask = (prices[ticker].index <= date) & (prices[ticker].index >= lookback_start)
        if sum(price_mask) == 0:
            continue  # Skip if no data in lookback period
            
        price_series = prices[ticker][price_mask]
        
        # Filter dividend series to the lookback period
        div_mask = (dividends[ticker].index <= date) & (dividends[ticker].index >= lookback_start)
        if sum(div_mask) == 0:
            continue  # Skip if no dividend data in lookback period
            
        div_series = dividends[ticker][div_mask]
        
        if not price_series.empty:
            # Calculate TTM dividend yield - sum all dividends over the past year
            ttm_div = div_series.sum() if not div_series.empty else 0
            current_price = price_series.iloc[-1]
            
            # Store dividend yield
            yields[ticker] = ttm_div / current_price if current_price > 0 else 0
            
    return yields


def calculate_volatilities(prices, date, lookback_days=180, min_data_points=60):
    """
    Calculate historical volatility for a set of stocks.
    
    Parameters:
    -----------
    prices : dict
        Dictionary mapping tickers to price Series
    date : datetime
        Date as of which to calculate volatility
    lookback_days : int, optional
        Number of days to look back for volatility calculation
    min_data_points : int, optional
        Minimum number of data points required for volatility calculation
        
    Returns:
    --------
    dict
        Dictionary mapping tickers to their annualized volatilities
    """
    volatilities = {}
    lookback_start = date - timedelta(days=lookback_days)
    
    for ticker in prices.keys():
        # Filter price series to the lookback period
        vol_mask = (prices[ticker].index <= date) & (prices[ticker].index >= lookback_start)
        if sum(vol_mask) < min_data_points:  # Need minimum data points
            continue
            
        vol_series = prices[ticker][vol_mask]
        
        # Calculate annualized volatility
        returns = vol_series.pct_change().dropna()
        volatilities[ticker] = returns.std() * np.sqrt(252)  # Annualize
        
    return volatilities


def select_stocks(prices, dividends, date, yield_threshold, volatility_percentile):
    """
    Select stocks based on dividend yield and volatility criteria.
    
    Parameters:
    -----------
    prices : dict
        Dictionary mapping tickers to price Series
    dividends : dict
        Dictionary mapping tickers to dividend Series
    date : datetime
        Selection date
    yield_threshold : float
        Minimum dividend yield required
    volatility_percentile : float
        Percentile cutoff for low volatility (e.g., 0.3 means bottom 30%)
        
    Returns:
    --------
    list
        List of selected ticker symbols
    dict
        Dictionary of yield values for selected tickers
    dict
        Dictionary of volatility values for selected tickers
    """
    # Calculate metrics for selection
    yields = calculate_dividend_yields(prices, dividends, date)
    volatilities = calculate_volatilities(prices, date)
    
    # Only consider stocks with valid data for both metrics
    valid_tickers = set(yields.keys()).intersection(set(volatilities.keys()))
    valid_tickers = [t for t in valid_tickers if volatilities[t] < np.inf]
    
    if not valid_tickers:
        return [], {}, {}
        
    # Sort by volatility and take bottom X%
    vol_df = pd.Series({t: volatilities[t] for t in valid_tickers}).sort_values()
    low_vol_count = max(1, int(len(vol_df) * volatility_percentile))
    low_vol_tickers = vol_df.iloc[:low_vol_count].index.tolist()
    
    # Filter for high yield among low volatility stocks
    selected = [t for t in low_vol_tickers if yields.get(t, 0) >= yield_threshold]
    
    # Create dictionaries of metrics for selected stocks
    selected_yields = {t: yields[t] for t in selected}
    selected_volatilities = {t: volatilities[t] for t in selected}
    
    return selected, selected_yields, selected_volatilities
