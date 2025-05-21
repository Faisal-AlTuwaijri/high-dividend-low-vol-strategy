"""
High-Dividend Low-Volatility Strategy: Parameter Optimization
===========================================================

This module contains functions for optimizing and testing different
parameter combinations for the high-dividend low-volatility strategy.
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import local modules
from .portfolio_simulation import simulate_portfolio
from .performance_metrics import calculate_performance_metrics, analyze_stock_selection
from .visualization import create_parameter_heatmap, create_performance_charts, plot_top_stocks


def create_parameter_grid(yield_thresholds, volatility_percentiles):
    """
    Create empty matrices for storing parameter optimization results.
    
    Parameters:
    -----------
    yield_thresholds : list
        List of yield threshold values to test
    volatility_percentiles : list
        List of volatility percentile values to test
        
    Returns:
    --------
    dict
        Dictionary containing empty matrices for storing results
    """
    # Create matrices for results
    cagr_matrix = np.zeros((len(yield_thresholds), len(volatility_percentiles)))
    sharpe_matrix = np.zeros((len(yield_thresholds), len(volatility_percentiles)))
    holdings_matrix = np.zeros((len(yield_thresholds), len(volatility_percentiles)))
    success_matrix = np.zeros((len(yield_thresholds), len(volatility_percentiles)), dtype=bool)
    
    return {
        'cagr_matrix': cagr_matrix,
        'sharpe_matrix': sharpe_matrix,
        'holdings_matrix': holdings_matrix,
        'success_matrix': success_matrix
    }


def find_optimal_parameters(matrices, yield_thresholds, volatility_percentiles, optimize_for='sharpe'):
    """
    Find the optimal parameters based on a specified metric.
    
    Parameters:
    -----------
    matrices : dict
        Dictionary containing result matrices from parameter optimization
    yield_thresholds : list
        List of yield threshold values tested
    volatility_percentiles : list
        List of volatility percentile values tested
    optimize_for : str, optional
        Metric to optimize for ('sharpe' or 'cagr')
        
    Returns:
    --------
    tuple
        (best_yield, best_vol, best_value)
    """
    if optimize_for.lower() == 'sharpe':
        matrix = matrices['sharpe_matrix']
        matrix_name = 'Sharpe Ratio'
    else:
        matrix = matrices['cagr_matrix']
        matrix_name = 'CAGR'
    
    # Apply success mask
    masked_matrix = np.where(matrices['success_matrix'], matrix, np.nan)
    
    if np.any(~np.isnan(masked_matrix)):
        max_idx = np.unravel_index(np.nanargmax(masked_matrix), masked_matrix.shape)
        best_yield = yield_thresholds[max_idx[0]]
        best_vol = volatility_percentiles[max_idx[1]]
        best_value = masked_matrix[max_idx]
        
        print(f"\nBest {matrix_name} of {best_value:.4f} achieved with:")
        print(f"  - Yield threshold: {best_yield:.1%}")
        print(f"  - Volatility percentile: {best_vol:.1%}")
        
        return best_yield, best_vol, best_value
    else:
        print(f"No valid {matrix_name} values found")
        return None, None, None


def run_parameter_optimization(prices, dividends, sp500_prices, risk_free,
                             yield_thresholds, volatility_percentiles,
                             start_date, end_date):
    """
    Run a parameter optimization for the high-dividend low-volatility strategy.
    
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
    yield_thresholds : list
        List of yield threshold values to test
    volatility_percentiles : list
        List of volatility percentile values to test
    start_date : str
        Start date for the backtest
    end_date : str
        End date for the backtest
        
    Returns:
    --------
    dict
        Results of the parameter optimization
    """
    # Set up parameter grid
    print("\nSetting up parameter grid...")
    total_combinations = len(yield_thresholds) * len(volatility_percentiles)
    print(f"Testing {total_combinations} parameter combinations")
    
    # Create matrices for results
    matrices = create_parameter_grid(yield_thresholds, volatility_percentiles)
    
    # Run all parameter combinations
    print("\nRunning parameter combinations...")
    overall_start_time = time.time()
    combination_count = 0
    
    for i, yield_threshold in enumerate(yield_thresholds):
        for j, volatility_percentile in enumerate(volatility_percentiles):
            combination_count += 1
            combination_start_time = time.time()
            
            print(f"\nRunning combination {combination_count}/{total_combinations}: "
                  f"Yield={yield_threshold:.1%}, Vol={volatility_percentile:.1%}")
            
            # Run the strategy for this parameter combination
            try:
                # Simulate the portfolio
                simulation_results = simulate_portfolio(
                    prices=prices,
                    dividends=dividends,
                    sp500_prices=sp500_prices,
                    risk_free=risk_free,
                    yield_threshold=yield_threshold,
                    volatility_percentile=volatility_percentile,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Calculate performance metrics
                portfolio_metrics, benchmark_metrics, aligned_portfolio, aligned_benchmark = calculate_performance_metrics(
                    simulation_results['portfolio_series'],
                    sp500_prices,
                    risk_free
                )
                
                if portfolio_metrics:
                    # Extract metrics
                    cagr = portfolio_metrics['CAGR']
                    sharpe = portfolio_metrics['Sharpe']
                    
                    # Store in matrices
                    matrices['cagr_matrix'][i, j] = cagr
                    matrices['sharpe_matrix'][i, j] = sharpe
                    
                    # Track holdings
                    if simulation_results['rebalance_history']:
                        last_holdings = len(simulation_results['rebalance_history'][-1]['selected'])
                        matrices['holdings_matrix'][i, j] = last_holdings
                    else:
                        matrices['holdings_matrix'][i, j] = 0
                    
                    # Mark as successful
                    matrices['success_matrix'][i, j] = True
                    
                    print(f"Result: CAGR={cagr:.2%}, Sharpe={sharpe:.2f}, "
                          f"Holdings={int(matrices['holdings_matrix'][i, j])}")
                else:
                    print("Failed to get valid performance metrics")
            except Exception as e:
                print(f"Error: {e}")
            
            # Calculate time metrics
            combination_time = time.time() - combination_start_time
            print(f"Combination completed in {combination_time:.1f} seconds")
            
            remaining_combinations = total_combinations - combination_count
            avg_time_per_combo = (time.time() - overall_start_time) / combination_count
            estimated_remaining = avg_time_per_combo * remaining_combinations
            
            print(f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")
    
    # Print overall runtime
    total_time = time.time() - overall_start_time
    print(f"\nTotal analysis time: {total_time/60:.1f} minutes")
    
    # Handle invalid results
    matrices['cagr_matrix'] = np.where(matrices['success_matrix'], matrices['cagr_matrix'], np.nan)
    matrices['sharpe_matrix'] = np.where(matrices['success_matrix'], matrices['sharpe_matrix'], np.nan)
    
    # Create heatmaps
    print("\nCreating heatmaps...")
    
    # Return results
    return {
        'matrices': matrices,
        'yield_thresholds': yield_thresholds,
        'volatility_percentiles': volatility_percentiles,
        'runtime': total_time
    }
