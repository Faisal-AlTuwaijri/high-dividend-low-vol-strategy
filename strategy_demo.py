"""
High-Dividend Low-Volatility Strategy - Example Notebook
========================================================

This notebook demonstrates the key functionality of the high-dividend
low-volatility backtesting framework, including running the strategy,
visualizing results, and analyzing performance.
"""

# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path for imports
sys.path.append('..')

# Import strategy modules
from src.data_utils import load_sp500_constituents, fetch_stock_data, fetch_risk_free_rate
from src.stock_selection import select_stocks
from src.portfolio_simulation import simulate_portfolio
from src.performance_metrics import calculate_performance_metrics, analyze_stock_selection
from src.visualization import (create_performance_metrics_table, create_performance_charts,
                             plot_top_stocks)
from main import analyze_high_dividend_strategy, display_strategy_metrics

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# File paths
CSV_PATH = "../data/sp500_historical_components.csv"
OUTPUT_DIR = "../results/notebook_demo"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create a shorter demo run
def run_quick_demo():
    """Run a quick demonstration of the backtesting framework with reduced parameters."""
    print("Running quick demonstration of high-dividend low-volatility strategy...")
    
    # Use a smaller parameter grid for faster execution
    yield_thresholds = [0.03, 0.04, 0.05]
    volatility_percentiles = [0.2, 0.3, 0.4]
    
    # Load data and run analysis
    sp500_tickers = load_sp500_constituents(CSV_PATH, "2014-12-31")
    
    # Add S&P 500 index for benchmark
    if '^GSPC' not in sp500_tickers:
        sp500_tickers.append('^GSPC')
    
    # Fetch a subset of tickers for the demo (first 100 plus S&P 500)
    demo_tickers = sp500_tickers[:100] + ['^GSPC']
    
    # Fetch stock data
    start_date = "2015-01-01"
    end_date = "2018-01-01"  # Shorter period for demo
    prices, dividends, valid_tickers, invalid_tickers = fetch_stock_data(
        demo_tickers, start_date, end_date
    )
    
    # Extract benchmark
    sp500_prices = prices['^GSPC']
    
    # Fetch risk-free rate
    risk_free = fetch_risk_free_rate(start_date, end_date)
    
    # Run for a single parameter combination
    yield_threshold = 0.03
    volatility_percentile = 0.3
    
    print(f"Running simulation with yield threshold = {yield_threshold:.1%} and volatility percentile = {volatility_percentile:.1%}")
    
    # Simulate portfolio
    simulation = simulate_portfolio(
        prices, dividends, sp500_prices, risk_free,
        yield_threshold, volatility_percentile,
        start_date, end_date
    )
    
    # Calculate metrics
    portfolio_metrics, benchmark_metrics, aligned_portfolio, aligned_benchmark = calculate_performance_metrics(
        simulation['portfolio_series'], sp500_prices, risk_free
    )
    
    # Display metrics table
    metrics_fig = create_performance_metrics_table(
        portfolio_metrics, benchmark_metrics, 
        f"Performance Metrics (Demo)"
    )
    metrics_fig.savefig(f"{OUTPUT_DIR}/demo_metrics.png", dpi=300, bbox_inches='tight')
    
    # Create performance charts
    perf_fig = create_performance_charts(
        aligned_portfolio, aligned_benchmark,
        portfolio_metrics, benchmark_metrics,
        simulation['actual_rebalance_dates'],
        "High-Div Low-Vol Strategy (Demo)",
        f"{OUTPUT_DIR}/demo_performance.png"
    )
    
    # Analyze stock selection
    stock_counts, stock_percentages, sorted_stocks = analyze_stock_selection(
        simulation['rebalance_history']
    )
    
    # Plot top stocks
    top_stocks_fig = plot_top_stocks(sorted_stocks, n=10, save_path=f"{OUTPUT_DIR}/demo_top_stocks.png")
    
    print("\nDemo Summary:")
    print(f"- CAGR: {portfolio_metrics['CAGR']*100:.2f}% vs {benchmark_metrics['CAGR']*100:.2f}% for S&P 500")
    print(f"- Volatility: {portfolio_metrics['Volatility']*100:.2f}% vs {benchmark_metrics['Volatility']*100:.2f}% for S&P 500")
    print(f"- Sharpe Ratio: {portfolio_metrics['Sharpe']:.2f}")
    print(f"- Max Drawdown: {portfolio_metrics['Max Drawdown']:.2f}% vs {benchmark_metrics['Max Drawdown']:.2f}% for S&P 500")
    
    return {
        'portfolio_metrics': portfolio_metrics,
        'benchmark_metrics': benchmark_metrics,
        'simulation': simulation
    }

# Run the demo
demo_results = run_quick_demo()

# # Uncomment to run full analysis (takes much longer)
# full_results = analyze_high_dividend_strategy(
#     csv_file_path=CSV_PATH,
#     target_date='2014-12-31',
#     start_date="2015-01-01", 
#     end_date="2024-01-01",
#     output_dir=OUTPUT_DIR
# )
# 
# # Display full metrics table
# display_strategy_metrics(full_results, save_path=f"{OUTPUT_DIR}/full_performance_summary.png")
