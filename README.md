# High-Dividend Low-Volatility Investment Strategy

A quantitative backtesting framework for a high-dividend low-volatility stock selection strategy with quarterly rebalancing.

## Strategy Overview
This strategy selects stocks based on:
- Dividend yields above a specified threshold
- Volatility below a specified percentile cutoff

## Key Features
- Parameter optimization across yield/volatility combinations
- Performance comparison against S&P 500 benchmark
- Comprehensive risk and return metrics
- Visualizations of results

## Performance Highlights
The strategy achieves similar returns to the S&P 500 (9.4% annualized) with:
- 47% lower volatility (9.5% vs 18.0%)
- 26% lower maximum drawdown (-25.2% vs -34.0%)

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- yfinance

## Usage
```python
# Run the analysis
results = analyze_high_dividend_strategy(
    csv_file_path="data/sp500_historical_components.csv",
    target_date='2014-12-31',
    start_date="2015-01-01",
    end_date="2024-01-01"
)
