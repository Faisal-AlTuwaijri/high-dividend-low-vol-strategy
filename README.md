# High-Dividend Low-Volatility Strategy Backtesting Framework

This repository contains a comprehensive backtesting framework for a high-dividend low-volatility stock selection strategy. The strategy aims to identify stocks with high dividend yields and low historical volatility, potentially providing attractive risk-adjusted returns.

## Strategy Overview

The high-dividend low-volatility strategy follows these key principles:

1. **Stock Selection Criteria**:
   - Stocks with dividend yields above a specified threshold
   - Stocks with volatility below a specified percentile cutoff
   
2. **Portfolio Construction**:
   - Equal weighting of selected stocks
   - Quarterly rebalancing

3. **Performance Analysis**:
   - Comparison against S&P 500 benchmark
   - Risk-adjusted return metrics (Sharpe ratio, volatility, drawdowns)
   - Parameter optimization to identify optimal threshold values

## Repository Structure

```
high-dividend-strategy/
│
├── data/
│   └── sp500_historical_components.csv
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py          # Data loading and preparation functions
│   ├── stock_selection.py     # Stock selection logic
│   ├── portfolio_simulation.py # Portfolio simulation
│   ├── performance_metrics.py  # Performance calculation
│   ├── visualization.py       # Visualization utilities
│   └── parameter_optimization.py # Parameter grid search
│
├── results/                   # Output directory for charts/analysis
│
├── README.md
├── requirements.txt
└── main.py                    # Main execution script
```

## Installation

1. Clone this repository:
```
git clone https://github.com/Faisal-AlTuwaijri/high-dividend-strategy.git
cd high-dividend-strategy
```

2. Install the required packages:
```
pip install -r requirements.txt
```

## Data Requirements

The framework requires a CSV file containing historical S&P 500 constituents. The file should contain at least:
- A `date` column with dates in YYYY-MM-DD format
- A `tickers` column with comma-separated lists of ticker symbols

Price and dividend data are fetched from Yahoo Finance using the `yfinance` library.

## Usage

### Basic Usage

Run the main analysis script:

```python
python main.py
```

This will:
1. Load S&P 500 constituents from the CSV file
2. Fetch historical price and dividend data
3. Run parameter optimization across yield/volatility combinations
4. Identify optimal parameters based on Sharpe ratio
5. Run detailed analysis of the optimal strategy
6. Generate visualizations in the `results/` directory

### Custom Analysis

To run with custom parameters:

```python
from main import analyze_high_dividend_strategy

results = analyze_high_dividend_strategy(
    csv_file_path="data/sp500_historical_components.csv",
    target_date="2014-12-31",   # Date for constituent selection
    start_date="2015-01-01",    # Backtest start date
    end_date="2024-01-01",      # Backtest end date
    output_dir="custom_results" # Output directory
)
```

### Display Performance Metrics

To generate a performance metrics table:

```python
from main import display_strategy_metrics

# After running the analysis
display_strategy_metrics(results, save_path="results/performance_summary.png")
```

## Key Features

1. **Efficient Data Handling**:
   - Fetches data once and reuses it across parameter combinations
   - Batch processing to avoid API rate limits
   - Proper handling of missing data and delisted stocks

2. **Modular Architecture**:
   - Separation of concerns with specialized modules
   - Easy to extend or modify individual components
   - Clean, well-documented code

3. **Comprehensive Analysis**:
   - Parameter optimization to find optimal thresholds
   - Detailed performance metrics including risk-adjusted returns
   - Stock selection analysis and concentration metrics

4. **Professional Visualizations**:
   - Performance comparison charts
   - Drawdown analysis
   - Parameter heatmaps
   - Most frequently selected stocks

## Example Results

The analysis produces several visualizations:

1. **Performance Comparison**:
   - Normalized performance vs. S&P 500
   - Drawdown comparison
   - Rolling 12-month returns

2. **Parameter Heatmaps**:
   - CAGR by parameter combination
   - Sharpe ratio by parameter combination
   - Number of holdings by parameter combination

3. **Performance Metrics Table**:
   - Total return
   - Annualized return (CAGR)
   - Volatility
   - Sharpe ratio
   - Maximum drawdown
   - Win rate
   - Downside capture

## Acknowledgments

- Yahoo Finance for providing historical price and dividend data
- S&P 500 historical constituents data compilation

## Contact

For questions or feedback, please contact altowaijri.faisal@gmail.com.
