"""
High-Dividend Low-Volatility Strategy: Visualization
==================================================

This module contains functions for visualizing the results of
the high-dividend low-volatility strategy backtest.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.table import Table
import matplotlib as mpl


def create_performance_metrics_table(portfolio_metrics, benchmark_metrics, title="PERFORMANCE METRICS"):
    """
    Create a professional-looking table comparing portfolio and benchmark metrics.
    
    Parameters:
    -----------
    portfolio_metrics : dict
        Dictionary containing portfolio performance metrics
    benchmark_metrics : dict
        Dictionary containing benchmark performance metrics
    title : str, optional
        Title for the table
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the table
    """
    # Set a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axis_off()
    
    # Table data
    table_data = [
        ["Metric", "Optimal Portfolio", "S&P500"],
        ["Total Return", f"{portfolio_metrics['Total Return']*100:.2f}%", f"{benchmark_metrics['Total Return']*100:.2f}%"],
        ["Annualized\nReturn", f"{portfolio_metrics['CAGR']*100:.1f}%", f"{benchmark_metrics['CAGR']*100:.1f}%"],
        ["Volatility", f"{portfolio_metrics['Volatility']*100:.1f}%", f"{benchmark_metrics['Volatility']*100:.1f}%"],
        ["Maximum\nDrawdown", f"{portfolio_metrics['Max Drawdown']:.1f}%", f"{benchmark_metrics['Max Drawdown']:.1f}%"],
    ]
    
    # Create Sharpe row if available
    if 'Sharpe' in portfolio_metrics:
        sharpe_row = ["Sharpe Ratio", f"{portfolio_metrics['Sharpe']:.2f}", "N/A"]
        table_data.insert(4, sharpe_row)  # Insert before Max Drawdown
    
    # Create additional metrics if available
    if 'Win Rate (Monthly)' in portfolio_metrics:
        win_rate_row = ["Win Rate\n(Monthly)", f"{portfolio_metrics['Win Rate (Monthly)']*100:.1f}%", "N/A"]
        table_data.append(win_rate_row)
    
    if 'Downside Capture' in portfolio_metrics:
        downside_row = ["Downside\nCapture", f"{portfolio_metrics['Downside Capture']:.2f}", "N/A"]
        table_data.append(downside_row)
    
    # Create the table
    the_table = ax.table(
        cellText=table_data[1:],  # Exclude header from cell text
        colLabels=table_data[0],   # Use first row as column labels
        loc='center',
        cellLoc='center',
        bbox=[0.1, 0.1, 0.8, 0.8]  # Position and size of the table
    )
    
    # Style the table
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    
    # Set header style
    for i, key in enumerate(table_data[0]):
        cell = the_table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#283858')  # Dark blue header
        cell.set_height(0.06)
    
    # Style the metric names column
    for i in range(1, len(table_data)):
        cell = the_table[(i-1, 0)]
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f0f0f8')  # Light blue-gray
    
    # Style other cells and highlight better performances
    for i in range(1, len(table_data)):
        # Get metric name (first column in current row)
        metric = table_data[i][0]
        
        # Format the portfolio value cell (second column)
        cell = the_table[(i-1, 1)]
        cell.set_height(0.06)
        
        # Format the benchmark value cell (third column)
        cell = the_table[(i-1, 2)]
        cell.set_height(0.06)
        
        # Highlight better performance based on metric type
        if metric in ["Total Return", "Annualized\nReturn", "Sharpe Ratio"]:
            # For these metrics, higher is better
            port_val = float(table_data[i][1].replace('%', ''))
            bench_val = float(table_data[i][2].replace('%', '').replace('N/A', '0'))
            
            if port_val > bench_val:
                the_table[(i-1, 1)].set_facecolor('#ebf5eb')  # Light green
            elif bench_val > port_val:
                the_table[(i-1, 2)].set_facecolor('#ebf5eb')  # Light green
        
        elif metric in ["Volatility", "Maximum\nDrawdown", "Downside\nCapture"]:
            # For these metrics, lower is better
            try:
                port_val = float(table_data[i][1].replace('%', ''))
                bench_val = float(table_data[i][2].replace('%', '').replace('N/A', '0'))
                
                if port_val < bench_val:
                    the_table[(i-1, 1)].set_facecolor('#ebf5eb')  # Light green
                elif bench_val < port_val:
                    the_table[(i-1, 2)].set_facecolor('#ebf5eb')  # Light green
            except ValueError:
                # Handle non-numeric or N/A values
                pass
    
    # Add a border to all cells
    for cell in the_table.get_celld().values():
        cell.set_linewidth(0.5)
        cell.set_edgecolor('#283858')  # Dark blue border
    
    # Add title
    ax.set_title(title, fontsize=20, fontweight='bold', color='#283858', pad=30)
    
    plt.tight_layout()
    
    return fig


def create_performance_charts(portfolio_series, benchmark_series, portfolio_metrics, 
                             benchmark_metrics, rebalance_dates, title, save_path=None):
    """
    Create performance visualization charts.
    
    Parameters:
    -----------
    portfolio_series : pandas.Series
        Daily portfolio values
    benchmark_series : pandas.Series
        Daily benchmark values
    portfolio_metrics : dict
        Dictionary of portfolio performance metrics
    benchmark_metrics : dict
        Dictionary of benchmark performance metrics
    rebalance_dates : list
        List of rebalance dates
    title : str
        Title for the charts
    save_path : str, optional
        File path to save the visualization
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the charts
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Performance chart (top left)
    ax1 = axs[0, 0]
    
    # Normalize to 100
    start_date = portfolio_series.index[0]
    start_value_portfolio = portfolio_series.iloc[0]
    start_value_benchmark = benchmark_series.iloc[0]
    
    norm_portfolio = portfolio_series / start_value_portfolio * 100
    norm_benchmark = benchmark_series / start_value_benchmark * 100
    
    ax1.plot(norm_portfolio.index, norm_portfolio, label=f"High-Div Low-Vol Strategy", linewidth=2)
    ax1.plot(norm_benchmark.index, norm_benchmark, label="S&P 500", linewidth=2)
    
    # Add title with date range
    start_year = portfolio_series.index[0].year
    end_year = portfolio_series.index[-1].year
    title_years = f"({start_year}-{end_year})"
    ax1.set_title(f"Performance Comparison {title_years}", fontsize=14)
    ax1.set_ylabel("Value (Indexed to 100)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Mark rebalance dates with vertical lines
    for date in rebalance_dates:
        ax1.axvline(x=date, color='gray', linestyle='--', alpha=0.3)
    
    # Add annotations for key metrics
    ax1.annotate(f"Strategy CAGR: {portfolio_metrics['CAGR']*100:.1f}%",
                 xy=(0.02, 0.95), xycoords='axes fraction', fontsize=12)
    ax1.annotate(f"S&P 500 CAGR: {benchmark_metrics['CAGR']*100:.1f}%",
                 xy=(0.02, 0.90), xycoords='axes fraction', fontsize=12)
    ax1.annotate(f"Volatility Ratio: {portfolio_metrics['Volatility']/benchmark_metrics['Volatility']:.2f}x",
                 xy=(0.02, 0.85), xycoords='axes fraction', fontsize=12)
    
    # 2. Drawdown chart (top right)
    ax2 = axs[0, 1]
    
    # Calculate drawdowns
    portfolio_drawdown = (portfolio_series / portfolio_series.cummax() - 1) * 100
    benchmark_drawdown = (benchmark_series / benchmark_series.cummax() - 1) * 100
    
    ax2.plot(portfolio_drawdown.index, portfolio_drawdown, label="Strategy Drawdown", linewidth=2)
    ax2.plot(benchmark_drawdown.index, benchmark_drawdown, label="S&P 500 Drawdown", linewidth=2)
    ax2.set_title("Drawdown Comparison", fontsize=14)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Display max drawdown
    ax2.annotate(f"Strategy Max DD: {portfolio_metrics['Max Drawdown']:.1f}%",
                 xy=(0.02, 0.95), xycoords='axes fraction', fontsize=12)
    ax2.annotate(f"S&P 500 Max DD: {benchmark_metrics['Max Drawdown']:.1f}%",
                 xy=(0.02, 0.90), xycoords='axes fraction', fontsize=12)
    
    # 3. Stock selection frequency (bottom left) - This will be filled by the caller
    ax3 = axs[1, 0]
    ax3.set_title("Most Frequently Selected Stocks", fontsize=14)
    ax3.set_xlabel("Selection Frequency (%)", fontsize=12)
    
    # 4. Rolling 12-month return comparison (bottom right)
    ax4 = axs[1, 1]
    
    if len(portfolio_series) >= 252:  # Need at least 1 year for rolling returns
        rolling_portfolio = portfolio_series.pct_change(periods=252).dropna() * 100
        rolling_benchmark = benchmark_series.pct_change(periods=252).dropna() * 100
        
        common_idx = rolling_portfolio.index.intersection(rolling_benchmark.index)
        if len(common_idx) > 0:
            ax4.plot(common_idx, rolling_portfolio.loc[common_idx], label="Strategy 12M Return", linewidth=2)
            ax4.plot(common_idx, rolling_benchmark.loc[common_idx], label="S&P 500 12M Return", linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title("Rolling 12-Month Returns", fontsize=14)
            ax4.set_ylabel("12-Month Return (%)", fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "Insufficient data for rolling returns",
                    horizontalalignment='center', verticalalignment='center')
    else:
        ax4.text(0.5, 0.5, "Need at least 1 year of data for rolling returns",
                horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved as: {save_path}")
    
    return fig


def create_parameter_heatmap(parameter_grid, metric_name, parameter_names, 
                            parameter_values, save_path=None):
    """
    Create a heatmap visualization for parameter optimization results.
    
    Parameters:
    -----------
    parameter_grid : numpy.ndarray
        Grid of metric values for each parameter combination
    metric_name : str
        Name of the metric being displayed (e.g., "CAGR (%)")
    parameter_names : tuple
        (row_param_name, col_param_name) names of parameters
    parameter_values : tuple
        (row_values, col_values) values for each parameter
    save_path : str, optional
        File path to save the heatmap
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the heatmap
    """
    plt.figure(figsize=(10, 8))
    
    # Format values for display
    if "CAGR" in metric_name and not "%" in metric_name:
        display_values = parameter_grid * 100  # Convert to percentage
        fmt = ".1f"
    else:
        display_values = parameter_grid
        fmt = ".2f"
    
    # Format row and column labels
    row_param_name, col_param_name = parameter_names
    row_values, col_values = parameter_values
    
    if "Yield" in row_param_name:
        row_labels = [f"{y:.1%}" for y in row_values]
    else:
        row_labels = [str(y) for y in row_values]
        
    if "Volatility" in col_param_name or "Percentile" in col_param_name:
        col_labels = [f"{v:.1%}" for v in col_values]
    else:
        col_labels = [str(v) for v in col_values]
    
    # Create heatmap
    ax = sns.heatmap(display_values, annot=True, fmt=fmt, cmap="YlGnBu",
                    xticklabels=col_labels, yticklabels=row_labels)
    
    ax.set_title(f"{metric_name}", fontsize=14)
    ax.set_xlabel(col_param_name, fontsize=12)
    ax.set_ylabel(row_param_name, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved as: {save_path}")
    
    return plt.gcf()


def plot_top_stocks(sorted_stocks, n=10, save_path=None):
    """
    Create a bar chart of the most frequently selected stocks.
    
    Parameters:
    -----------
    sorted_stocks : list
        List of (ticker, frequency) tuples, sorted by frequency
    n : int, optional
        Number of top stocks to display
    save_path : str, optional
        File path to save the visualization
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the chart
    """
    plt.figure(figsize=(10, 8))
    
    # Limit to top N stocks
    top_n = min(n, len(sorted_stocks))
    tickers = [t[0] for t in sorted_stocks[:top_n]]
    frequencies = [t[1] for t in sorted_stocks[:top_n]]
    
    # Create horizontal bar chart
    plt.barh(tickers, frequencies)
    plt.xlabel("Selection Frequency (%)", fontsize=12)
    plt.title("Most Frequently Selected Stocks", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved as: {save_path}")
    
    return plt.gcf()


def plot_sector_breakdown(sector_weights, title="Sector Breakdown", save_path=None):
    """
    Create a pie chart showing the sector breakdown of the portfolio.
    
    Parameters:
    -----------
    sector_weights : dict
        Dictionary mapping sectors to their weights
    title : str, optional
        Title for the chart
    save_path : str, optional
        File path to save the visualization
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the chart
    """
    plt.figure(figsize=(10, 8))
    
    # Sort sectors by weight
    sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
    labels = [f"{s[0]}: {s[1]:.1f}%" for s in sorted_sectors]
    values = [s[1] for s in sorted_sectors]
    
    # Create pie chart
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Sector breakdown chart saved as: {save_path}")
    
    return plt.gcf()
