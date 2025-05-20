"""
High-Dividend Low-Volatility Strategy: Performance Metrics
=========================================================

This module contains functions for calculating and analyzing 
performance metrics for the backtested strategy.
"""

import pandas as pd
import numpy as np


def calculate_performance_metrics(portfolio_series, benchmark_series, risk_free_series):
    """
    Calculate performance metrics for a portfolio.
    
    Parameters:
    -----------
    portfolio_series : pandas.Series
        Daily portfolio values
    benchmark_series : pandas.Series
        Daily benchmark values (e.g., S&P 500)
    risk_free_series : pandas.Series
        Daily risk-free rate
        
    Returns:
    --------
    tuple
        (portfolio_metrics, benchmark_metrics, aligned_portfolio, aligned_benchmark)
        Dictionaries containing performance metrics and aligned series
    """
    # Handle NaN values
    if portfolio_series.isna().any():
        print(f"Warning: Portfolio series contains {portfolio_series.isna().sum()} NaN values")
        portfolio_series = portfolio_series.ffill()
    
    # Align indices
    aligned_indices = benchmark_series.index.intersection(portfolio_series.index)
    
    if len(aligned_indices) < 252:  # Require at least one year of data
        print("Warning: Not enough data for performance metrics (need at least 1 year)")
        return {}, {}, None, None
        
    aligned_portfolio = portfolio_series.loc[aligned_indices]
    aligned_benchmark = benchmark_series.loc[aligned_indices]
    
    # Calculate years
    years = len(aligned_portfolio) / 252
    
    # Total returns
    total_return = (aligned_portfolio.iloc[-1] / aligned_portfolio.iloc[0]) - 1
    benchmark_total_return = (aligned_benchmark.iloc[-1] / aligned_benchmark.iloc[0]) - 1
    
    # CAGRs
    cagr = (1 + total_return) ** (1 / years) - 1
    benchmark_cagr = (1 + benchmark_total_return) ** (1 / years) - 1
    
    # Calculate returns
    returns = aligned_portfolio.pct_change().dropna()
    benchmark_returns = aligned_benchmark.pct_change().dropna()
    
    # Volatilities
    volatility = returns.std() * np.sqrt(252)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    rf_aligned = risk_free_series.reindex(returns.index, method='ffill')
    excess_returns = returns - rf_aligned
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Drawdowns
    portfolio_drawdown = (aligned_portfolio / aligned_portfolio.cummax() - 1) * 100
    benchmark_drawdown = (aligned_benchmark / aligned_benchmark.cummax() - 1) * 100
    
    max_drawdown = portfolio_drawdown.min()
    max_dd_date = portfolio_drawdown.idxmin()
    max_benchmark_dd = benchmark_drawdown.min()
    
    # Monthly metrics
    monthly_portfolio = aligned_portfolio.resample('ME').last().pct_change(fill_method=None).dropna()
    monthly_benchmark = aligned_benchmark.resample('ME').last().pct_change(fill_method=None).dropna()
    
    # Win rate
    common_months = monthly_portfolio.index.intersection(monthly_benchmark.index)
    months_won = sum(monthly_portfolio.loc[common_months] > monthly_benchmark.loc[common_months])
    win_rate = months_won / len(common_months) if len(common_months) > 0 else 0
    
    # Downside capture
    down_months = monthly_benchmark[monthly_benchmark < 0].index
    if len(down_months) > 0:
        downside_capture = (1 + monthly_portfolio.loc[down_months]).prod() / (1 + monthly_benchmark.loc[down_months]).prod()
    else:
        downside_capture = 0
    
    # Sortino ratio (use 0 as minimum acceptable return)
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0001
    sortino = excess_returns.mean() * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown duration
    is_in_drawdown = portfolio_drawdown < 0
    drawdown_start = pd.Series(is_in_drawdown.index, index=is_in_drawdown.index)
    drawdown_start[~is_in_drawdown] = pd.NaT
    drawdown_start = drawdown_start.ffill()
    
    if drawdown_start.notna().any() and is_in_drawdown.iloc[-1]:
        # Add the last date to measure ongoing drawdowns
        drawdown_start = pd.concat([drawdown_start, pd.Series([drawdown_start.iloc[-1]], index=[is_in_drawdown.index[-1] + pd.Timedelta(days=1)])])
    
    drawdown_duration = drawdown_start.index - drawdown_start
    max_dd_duration = drawdown_duration.max().days if not drawdown_duration.empty else 0
    
    # Create metrics dictionaries
    portfolio_metrics = {
        "Total Return": float(total_return),
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Max Drawdown": float(max_drawdown),
        "Max Drawdown Date": max_dd_date,
        "Max DD Duration (days)": max_dd_duration,
        "Volatility": float(volatility),
        "Win Rate (Monthly)": float(win_rate),
        "Downside Capture": float(downside_capture)
    }
    
    benchmark_metrics = {
        "Total Return": float(benchmark_total_return),
        "CAGR": float(benchmark_cagr),
        "Volatility": float(benchmark_volatility),
        "Max Drawdown": float(max_benchmark_dd)
    }
    
    return portfolio_metrics, benchmark_metrics, aligned_portfolio, aligned_benchmark


def analyze_stock_selection(rebalance_history):
    """
    Analyze the frequency and distribution of stock selections.
    
    Parameters:
    -----------
    rebalance_history : list
        List of dictionaries containing rebalance information
        
    Returns:
    --------
    tuple
        (stock_counts, stock_percentages, sorted_stocks)
    """
    # Count occurrences of each stock
    stock_counts = {}
    for rb in rebalance_history:
        for ticker in rb['selected']:
            stock_counts[ticker] = stock_counts.get(ticker, 0) + 1
    
    # Calculate percentage of total rebalances
    total_rebalances = len(rebalance_history)
    stock_percentages = {ticker: count/total_rebalances*100 for ticker, count in stock_counts.items()}
    
    # Sort by frequency
    sorted_stocks = sorted(stock_percentages.items(), key=lambda x: x[1], reverse=True)
    
    return stock_counts, stock_percentages, sorted_stocks


def calculate_sector_exposure(selected_tickers, date=None):
    """
    Calculate sector exposure for a portfolio of stocks.
    
    Parameters:
    -----------
    selected_tickers : list
        List of selected ticker symbols
    date : datetime, optional
        Date for which to get sector data (for historical analysis)
        
    Returns:
    --------
    dict
        Dictionary mapping sectors to their weights
    """
    try:
        import yfinance as yf
        
        # Get sector information for each ticker
        sector_data = {}
        for ticker in selected_tickers:
            try:
                info = yf.Ticker(ticker).info
                sector = info.get('sector', 'Unknown')
                sector_data[ticker] = sector
            except:
                sector_data[ticker] = 'Unknown'
        
        # Calculate sector weights (equal weight assumption)
        sector_counts = {}
        for ticker, sector in sector_data.items():
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Convert to percentages
        total_tickers = len(selected_tickers)
        sector_weights = {sector: count/total_tickers*100 for sector, count in sector_counts.items()}
        
        return sector_weights
    
    except ImportError:
        print("yfinance is required for sector analysis")
        return {}
