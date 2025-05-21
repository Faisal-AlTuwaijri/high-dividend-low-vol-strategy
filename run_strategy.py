"""
High-Dividend Low-Volatility Strategy Runner
===========================================

This script handles all path issues and runs the high-dividend strategy analysis.
"""

import os
import sys
import importlib.util
import glob

def find_src_directory():
    """Find the src directory by searching from the current directory upward."""
    current_dir = os.path.abspath(os.getcwd())
    
    # First, try to find src in the current directory or subdirectories
    for root, dirs, _ in os.walk(current_dir):
        if 'src' in dirs:
            src_path = os.path.join(root, 'src')
            # Verify it contains our expected modules
            if any(glob.glob(os.path.join(src_path, "*.py"))):
                return src_path
    
    print("Could not find src directory with the required modules.")
    sys.exit(1)

def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def setup_environment():
    """Set up the Python environment for running the strategy."""
    src_dir = find_src_directory()
    print(f"Found src directory at: {src_dir}")
    
    # Add src directory to Python path
    sys.path.insert(0, os.path.dirname(src_dir))
    
    # Import all required modules directly from their file paths
    modules = {}
    for module_file in glob.glob(os.path.join(src_dir, "*.py")):
        if os.path.basename(module_file) == "__init__.py":
            continue
            
        module_name = os.path.splitext(os.path.basename(module_file))[0]
        modules[module_name] = import_module_from_file(module_name, module_file)
        print(f"Imported {module_name}")
    
    return modules, src_dir

def find_data_file(base_dir):
    """Find the sp500_historical_components.csv file."""
    # Look for the data directory and file
    for root, dirs, files in os.walk(base_dir):
        if 'data' in dirs:
            data_dir = os.path.join(root, 'data')
            csv_path = os.path.join(data_dir, 'sp500_historical_components.csv')
            if os.path.exists(csv_path):
                return csv_path
    
    # If not found, check if it's in the current directory
    if os.path.exists('sp500_historical_components.csv'):
        return 'sp500_historical_components.csv'
        
    return None

def create_output_dir(base_dir):
    """Create an output directory for results."""
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def run_strategy(modules, csv_file, output_dir):
    """Run the high-dividend low-volatility strategy."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract needed functions from modules
    load_sp500_constituents = modules['data_utils'].load_sp500_constituents
    fetch_stock_data = modules['data_utils'].fetch_stock_data
    fetch_risk_free_rate = modules['data_utils'].fetch_risk_free_rate
    select_stocks = modules['stock_selection'].select_stocks
    simulate_portfolio = modules['portfolio_simulation'].simulate_portfolio
    calculate_performance_metrics = modules['performance_metrics'].calculate_performance_metrics
    analyze_stock_selection = modules['performance_metrics'].analyze_stock_selection
    create_performance_metrics_table = modules['visualization'].create_performance_metrics_table
    create_performance_charts = modules['visualization'].create_performance_charts
    create_parameter_heatmap = modules['visualization'].create_parameter_heatmap
    plot_top_stocks = modules['visualization'].plot_top_stocks
    run_parameter_optimization = modules['parameter_optimization'].run_parameter_optimization
    find_optimal_parameters = modules['parameter_optimization'].find_optimal_parameters
    
    # Define analysis parameters
    target_date = '2014-12-31'
    start_date = '2015-01-01'
    end_date = '2018-01-01'  # Shorter for demo, use '2024-01-01' for full analysis
    
    print(f"\nRunning high-dividend strategy analysis:")
    print(f"CSV file: {csv_file}")
    print(f"Target date: {target_date}")
    print(f"Analysis period: {start_date} to {end_date}")
    print(f"Results will be saved to: {output_dir}")
    
    # Step 1: Load constituents
    sp500_tickers = load_sp500_constituents(csv_file, target_date)
    if not sp500_tickers:
        print("Error: Could not load S&P 500 constituents.")
        return
        
    # Add S&P 500 index for benchmark
    if '^GSPC' not in sp500_tickers:
        sp500_tickers.append('^GSPC')
    
    # Step 2: Fetch data for tickers
    prices, dividends, valid_tickers, invalid_tickers = fetch_stock_data(
        sp500_tickers, start_date, end_date
    )
    
    # Make sure we have the benchmark
    if '^GSPC' not in valid_tickers:
        print("Error: Could not fetch S&P 500 index data (^GSPC)")
        return
        
    # Extract S&P 500 benchmark
    sp500_prices = prices['^GSPC']
    
    # Fetch risk-free rate data
    risk_free = fetch_risk_free_rate(start_date, end_date)
    
    # Step 3: Set up parameter grid
    print("\nSetting up parameter grid...")
    yield_thresholds = [0.02, 0.03, 0.04]  # Reduced for demo
    volatility_percentiles = [0.2, 0.3, 0.4]  # Reduced for demo
    
    # Step 4: Run parameter optimization
    opt_results = run_parameter_optimization(
        prices, dividends, sp500_prices, risk_free,
        yield_thresholds, volatility_percentiles,
        start_date, end_date
    )
    
    matrices = opt_results['matrices']
    
    # Find the best parameter combination (optimizing for Sharpe by default)
    best_yield, best_vol, _ = find_optimal_parameters(
        matrices, yield_thresholds, volatility_percentiles, 'sharpe'
    )
    
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
        metrics_table_file = os.path.join(output_dir, "performance_metrics.png")
        metrics_fig = create_performance_metrics_table(
            optimal_portfolio_metrics, 
            optimal_benchmark_metrics,
            f"PERFORMANCE METRICS (Y={best_yield:.1%}, V={best_vol:.1%})"
        )
        metrics_fig.savefig(metrics_table_file, dpi=300, bbox_inches='tight')
        print(f"Performance metrics table saved to: {metrics_table_file}")
        
        # Create performance charts
        optimal_file = os.path.join(output_dir, "optimal_strategy.png")
        create_performance_charts(
            aligned_portfolio,
            aligned_benchmark,
            optimal_portfolio_metrics,
            optimal_benchmark_metrics,
            optimal_simulation['actual_rebalance_dates'],
            f"Optimal High-Div Low-Vol Strategy",
            optimal_file
        )
        
        # Analyze stock selection
        if optimal_simulation['rebalance_history']:
            stock_counts, stock_percentages, sorted_stocks = analyze_stock_selection(
                optimal_simulation['rebalance_history']
            )
            
            # Create stock frequency chart
            freq_file = os.path.join(output_dir, "stock_frequency.png")
            plot_top_stocks(sorted_stocks, n=15, save_path=freq_file)
            
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Display a summary of results
        print("\nStrategy Performance Summary:")
        print(f"- CAGR: {optimal_portfolio_metrics['CAGR']*100:.2f}% vs {optimal_benchmark_metrics['CAGR']*100:.2f}% for S&P 500")
        print(f"- Volatility: {optimal_portfolio_metrics['Volatility']*100:.2f}% vs {optimal_benchmark_metrics['Volatility']*100:.2f}% for S&P 500")
        print(f"- Sharpe Ratio: {optimal_portfolio_metrics['Sharpe']:.2f}")
        print(f"- Max Drawdown: {optimal_portfolio_metrics['Max Drawdown']:.2f}% vs {optimal_benchmark_metrics['Max Drawdown']:.2f}% for S&P 500")
        
        return True
    else:
        print("Error: Could not find optimal parameters.")
        return False

def main():
    """Main entry point for the script."""
    print("High-Dividend Low-Volatility Strategy Analysis")
    print("=============================================")
    
    # Set up the environment
    modules, base_dir = setup_environment()
    
    # Find the data file
    csv_file = find_data_file(base_dir)
    if not csv_file:
        print("Error: Could not find sp500_historical_components.csv")
        print("Please ensure the file is in the data directory or current directory.")
        return
    
    # Create output directory
    output_dir = create_output_dir(base_dir)
    
    # Run the strategy
    success = run_strategy(modules, csv_file, output_dir)
    
    if success:
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis completed with errors.")

if __name__ == "__main__":
    main()
