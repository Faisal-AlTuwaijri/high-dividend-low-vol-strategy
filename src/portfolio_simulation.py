"""
High-Dividend Low-Volatility Strategy: Portfolio Simulation
=========================================================

This module contains functions for simulating a portfolio using 
the high-dividend low-volatility strategy.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def simulate_portfolio(prices, dividends, sp500_prices, risk_free, 
                      yield_threshold, volatility_percentile,
                      start_date, end_date, rebalance_freq="QE",
                      stock_selection_func=None):
    """
    Simulate a portfolio using the high-dividend low-volatility strategy.
    
    Parameters:
    -----------
    prices : dict
        Dictionary mapping tickers to price Series
    dividends : dict
        Dictionary mapping tickers to dividend Series
    sp500_prices : pandas.Series
        S&P 500 price series for benchmark comparison
    risk_free : pandas.Series
        Risk-free rate series
    yield_threshold : float
        Minimum dividend yield required
    volatility_percentile : float
        Percentile cutoff for low volatility
    start_date : str
        Start date for the simulation
    end_date : str
        End date for the simulation
    rebalance_freq : str, optional
        Pandas frequency string for rebalancing (default: "QE" for quarter-end)
    stock_selection_func : function, optional
        Function to use for stock selection. If None, the select_stocks function
        from stock_selection.py is used.
        
    Returns:
    --------
    dict
        Dictionary containing simulation results
    """
    # Import here to avoid circular imports
    from stock_selection import select_stocks
    
    # Use provided selection function or default to select_stocks
    if stock_selection_func is None:
        stock_selection_func = select_stocks
    
    # Create a ticker list excluding the S&P 500 index
    stock_universe = [ticker for ticker in prices.keys() if ticker != '^GSPC']
    
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Set rebalance dates with specified frequency
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
    print(f"Scheduled {len(rebalance_dates)} rebalance dates with {rebalance_freq} frequency")
    
    # Create date range for daily portfolio values
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    
    # Initialize portfolio tracking variables
    initial_capital = 1000000  # $1M
    portfolio_value = [initial_capital]
    cash = initial_capital
    holdings = {}  # Dictionary to track {ticker: shares}
    last_value = initial_capital
    rebalance_history = []
    actual_rebalance_dates = []
    
    # Simulate the portfolio
    for date in dates:
        # Check if this is a valid trading date (S&P 500 has data)
        if date not in sp500_prices.index:
            portfolio_value.append(last_value)
            continue
        
        # Check if we need to rebalance today
        is_rebalance_day = any(rd.strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d') for rd in rebalance_dates)
        
        if is_rebalance_day:
            actual_rebalance_dates.append(date)
            print(f"Rebalancing on {date.strftime('%Y-%m-%d')}")
            
            # Select stocks based on criteria
            selected, selected_yields, selected_volatilities = stock_selection_func(
                prices, dividends, date, yield_threshold, volatility_percentile
            )
            
            # Log selected stocks and their metrics
            print(f"  Selected {len(selected)} stocks: {', '.join(selected[:5])}{', ...' if len(selected) > 5 else ''}")
            for ticker in selected[:5]:  # Limit to first 5 to avoid excessive output
                print(f"    {ticker}: Yield={selected_yields[ticker]:.2%}, Vol={selected_volatilities[ticker]:.2%}")
            if len(selected) > 5:
                print(f"    ... and {len(selected)-5} more")
            
            # Sell existing positions to get cash
            for ticker, shares in holdings.items():
                if ticker in prices and date in prices[ticker].index:
                    sale_value = shares * prices[ticker][date]
                    cash += sale_value
            
            # Reset holdings
            holdings = {}
            
            # Allocate to new selections
            if selected:
                weight = 1.0 / len(selected)
                for ticker in selected:
                    if ticker in prices and date in prices[ticker].index:
                        allocation = weight * cash
                        price = prices[ticker][date]
                        shares = allocation / price
                        holdings[ticker] = shares
                        cash -= shares * price
            
            # Record rebalance details
            portfolio_total = sum(holdings.get(t, 0) * prices[t][date] for t in holdings if t in prices and date in prices[t].index) + cash
            rebalance_history.append({
                'date': date,
                'selected': selected,
                'portfolio_value': portfolio_total,
                'yields': selected_yields,
                'volatilities': selected_volatilities
            })
        
        # Calculate daily portfolio value
        daily_value = cash
        for ticker, shares in holdings.items():
            if ticker in prices and date in prices[ticker].index:
                daily_value += shares * prices[ticker][date]
            else:
                # Use last known price if today's is missing
                if ticker in prices:
                    last_price = prices[ticker][prices[ticker].index < date]
                    if not last_price.empty:
                        daily_value += shares * last_price.iloc[-1]
        
        # Add dividends to cash
        for ticker, shares in holdings.items():
            if ticker in dividends and date in dividends[ticker].index:
                dividend_per_share = dividends[ticker][date]
                if dividend_per_share > 0:
                    dividend_income = shares * dividend_per_share
                    cash += dividend_income
        
        # Update portfolio value
        if not np.isnan(daily_value) and daily_value > 0:
            portfolio_value.append(daily_value)
            last_value = daily_value
        else:
            portfolio_value.append(last_value)  # Use last valid value
    
    # Convert to Series for easier analysis
    portfolio_series = pd.Series(portfolio_value, index=[dates[0] - pd.Timedelta(days=1)] + list(dates[:len(portfolio_value)-1]))
    
    return {
        'portfolio_series': portfolio_series,
        'rebalance_history': rebalance_history,
        'actual_rebalance_dates': actual_rebalance_dates,
        'initial_capital': initial_capital,
        'final_holdings': holdings,
        'cash': cash
    }
