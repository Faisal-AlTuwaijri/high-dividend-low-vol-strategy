"""
High-Dividend Low-Volatility Strategy Backtesting Framework
==========================================================

This module implements a backtesting framework for a high-dividend low-volatility
stock selection strategy with quarterly rebalancing.

The strategy selects stocks with:
1. Dividend yields above a specified threshold
2. Volatility below a specified percentile cutoff
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# Import local modules
from src.data_utils import load_sp500_constituents, fetch_stock_data, fetch_risk_free_rate
from src.stock_selection import select_stocks
from src.portfolio_simulation import simulate_portfolio
from src.performance_metrics import calculate_performance_metrics, analyze_stock_selection
from src.visualization import (create_performance_metrics_table, create_performance_charts,
                         create_parameter_heatmap, plot_top_stocks, plot_sector_breakdown)
from src.parameter_optimization import (create_parameter_grid, find_optimal_parameters, 
                                  run_parameter_optimization)


def analyze_high_dividend_strategy(csv_file_path, target_date='2014-12-31', 
                                 start_date='2015-01-01', end_date='2024-01-01',
                                 output_dir='results'):
    """
    Analyze a high-dividend low-volatility strategy with parameter optimization.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing historical S&P 500 constituents
    target_date : str
        Date for which to get the S&P 500 constituents (before backtest start)
    start_date : str
        Start date for the backtest
    end_date : str
        End date for the backtest
    output_dir : str
        Directory to save results and visualizations
        
    Returns:
    --------
    dict
        Results of the parameter sensitivity analysis
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Step 1: Load S&P 500 constituents
    sp500_tickers = load_sp500_constituents(csv_file_path, target_date)
    if not sp500_tickers:
        return None
        
    # Add S&P 500 index for benchmark
    if '^GSPC' not in sp500_tickers:
        sp500_tickers.append('^GSPC')
    
    # Step 2: Fetch data for all tickers
    prices, dividends, valid_tickers, invalid_tickers = fetch_stock_data(
        sp500_tickers, start_date, end_date
    )
    
    # Make sure we have the benchmark
    if '^GSPC' not in valid_tickers:
        print("Error: Could not fetch S&P 500 index data (^GSPC)")
        return None
        
    # Extract S&P 500 benchmark
    sp500_prices = prices['^GSPC']
    
    # Fetch risk-free rate data
    risk_free = fetch_risk_free_rate(start_date, end_date)
    
    # Step 3: Set up parameter grid
    print("\nSetting up parameter grid...")
    yield_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06]
    volatility_percentiles = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Step 4: Run parameter optimization
    opt_results = run_parameter_optimization(
        prices, dividends, sp500_prices, risk_free,
        yield_thresholds, volatility_percentiles,
        start_date, end_date
    )
    
    matrices = opt_results['matrices']
    
    # Create and save heatmaps
    target_date_str = pd.to_datetime(target_date).strftime('%Y%m%d')
    
    # CAGR heatmap
    cagr_file = os.path.join(output_dir, f"parameter_heatmap_CAGR_{target_date_str}.png")
    create_parameter_heatmap(
        matrices['cagr_matrix'] * 100,  # Convert to percentage
        f"CAGR (%) - S&P 500 ({target_date})",
        ("Yield Threshold", "Volatility Percentile"),
        (yield_thresholds, volatility_percentiles),
        cagr_file
    )
    
    # Sharpe heatmap
    sharpe_file = os.path.join(output_dir, f"parameter_heatmap_Sharpe_{target_date_str}.png")
    create_parameter_heatmap(
        matrices['sharpe_matrix'],
        f"Sharpe Ratio - S&P 500 ({target_date})",
        ("Yield Threshold", "Volatility Percentile"),
        (yield_thresholds, volatility_percentiles),
        sharpe_file
    )
    
    # Holdings heatmap
    holdings_file = os.path.join(output_dir, f"holdings_heatmap_SP500_{target_date_str}.png")
    create_parameter_heatmap(
        matrices['holdings_matrix'],
        f"Number of Holdings - S&P 500 ({target_date})",
        ("Yield Threshold", "Volatility Percentile"),
        (yield_thresholds, volatility_percentiles),
        holdings_file
    )
    
    # Find the best parameter combination (optimizing for Sharpe by default)
    optimize_for_sharpe = True  # Set to False to optimize for CAGR instead
    
    if optimize_for_sharpe:
        best_yield, best_vol, _ = find_optimal_parameters(
            matrices, yield_thresholds, volatility_percentiles, 'sharpe'
        )
    else:
        best_yield, best_vol, _ = find_optimal_parameters(
            matrices, yield_thresholds, volatility_percentiles, 'cagr'
        )
    
    optimal_simulation = None
    optimal_portfolio_metrics = None
    optimal_benchmark_metrics = None
    
    # Run with optimal parameters if we found a valid combination
    if best_yield is not None and best_vol is not None:
        print("\nRunning strategy with optimal parameters...")
        
        # Run the optimal strategy
        optimal_simulation = simulate_portfolio(
            prices=prices,
            dividends=dividends,
            sp500_prices=sp500_prices,
            risk_free=risk_free,
            yield_threshold=best_yield,
            volatility_percentile=best_vol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate performance metrics
        optimal_portfolio_metrics, optimal_benchmark_metrics, aligned_portfolio, aligned_benchmark = calculate_performance_metrics(
            optimal_simulation['portfolio_series'],
            sp500_prices,
            risk_free
        )
        
        # Create performance metrics table
        metrics_table_file = os.path.join(output_dir, f"performance_metrics_{target_date_str}.png")
        metrics_fig = create_performance_metrics_table(
            optimal_portfolio_metrics, 
            optimal_benchmark_metrics,
            f"PERFORMANCE METRICS (Y={best_yield:.1%}, V={best_vol:.1%})"
        )
        metrics_fig.savefig(metrics_table_file, dpi=300, bbox_inches='tight')
        print(f"Performance metrics table saved to: {metrics_table_file}")
        
        # Analyze stock selection
        if optimal_simulation['rebalance_history']:
            stock_counts, stock_percentages, sorted_stocks = analyze_stock_selection(
                optimal_simulation['rebalance_history']
            )
            
            print("\nStock Selection Distribution:")
            print(f"Total unique stocks selected: {len(stock_counts)}")
            print(f"Most frequently selected stocks (top 15):")
            for ticker, percentage in sorted_stocks[:15]:
                print(f"  {ticker}: {percentage:.1f}% of periods")
            
            # Calculate concentration metrics
            top_5_pct = sum(sorted_stocks[i][1] for i in range(min(5, len(sorted_stocks)))) / 100
            top_10_pct = sum(sorted_stocks[i][1] for i in range(min(10, len(sorted_stocks)))) / 100
            
            print(f"\nConcentration metrics:")
            print(f"  Top 5 stocks account for {top_5_pct:.1%} of all selections")
            print(f"  Top 10 stocks account for {top_10_pct:.1%} of all selections")
            
            # Create stock frequency chart
            freq_file = os.path.join(output_dir, f"stock_frequency_{target_date_str}.png")
            plot_top_stocks(sorted_stocks, n=15, save_path=freq_file)
        
        # Create visualizations for optimal strategy
        optimal_file = os.path.join(output_dir, f"optimal_strategy_y{int(best_yield*100)}_v{int(best_vol*100)}.png")
        create_performance_charts(
            aligned_portfolio,
            aligned_benchmark,
            optimal_portfolio_metrics,
            optimal_benchmark_metrics,
            optimal_simulation['actual_rebalance_dates'],
            f"Optimal High-Div Low-Vol Strategy (Y={best_yield:.1%}, V={best_vol:.1%})",
            optimal_file
        )
    
    # Return the results
    return {
        'cagr_matrix': matrices['cagr_matrix'],
        'sharpe_matrix': matrices['sharpe_matrix'],
        'holdings_matrix': matrices['holdings_matrix'],
        'yield_thresholds': yield_thresholds,
        'volatility_percentiles': volatility_percentiles,
        'valid_tickers': valid_tickers,
        'invalid_tickers': invalid_tickers,
        'optimal_results': optimal_simulation,
        'portfolio_metrics': optimal_portfolio_metrics,
        'sp500_metrics': optimal_benchmark_metrics,
        'best_parameters': {
            'yield_threshold': best_yield,
            'volatility_percentile': best_vol
        } if best_yield is not None else None
    }


def display_strategy_metrics(results_dict, save_path=None):
    """
    Display key metrics from a completed strategy analysis.
    
    Parameters:
    -----------
    results_dict : dict
        Results dictionary from analyze_high_dividend_strategy
    save_path : str, optional
        File path to save the metrics table
    
    Returns:
    --------
    matplotlib.figure.Figure
        The metrics table figure
    """
    if not results_dict or 'portfolio_metrics' not in results_dict or not results_dict['portfolio_metrics']:
        print("Error: No portfolio metrics found in the provided results dictionary.")
        return None
    
    # Extract the metrics
    portfolio_metrics = results_dict['portfolio_metrics']
    benchmark_metrics = results_dict['sp500_metrics']
    
    if not portfolio_metrics or not benchmark_metrics:
        print("Error: Missing portfolio or benchmark metrics in results.")
        return None
    
    # Create and display the metrics table
    title = "PERFORMANCE METRICS"
    if 'best_parameters' in results_dict and results_dict['best_parameters']:
        params = results_dict['best_parameters']
        title += f" (Y={params['yield_threshold']:.1%}, V={params['volatility_percentile']:.1%})"
        
    fig = create_performance_metrics_table(portfolio_metrics, benchmark_metrics, title)
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics table saved to: {save_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Assuming the CSV file is in the current directory
    csv_file_path = "data/sp500_historical_components.csv"  # Update with your actual file path
    
    # Run the optimized analysis
    results = analyze_high_dividend_strategy(
        csv_file_path=csv_file_path,
        target_date='2014-12-31',
        start_date="2015-01-01",
        end_date="2024-01-01",
        output_dir="results"
    )
    
    # Display metrics table
    if results:
        display_strategy_metrics(results, save_path="results/performance_summary.png")
