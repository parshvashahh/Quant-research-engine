"""Quantitative Research Engine
Author      : Parshva Shah
LinkedIn    : https://www.linkedin.com/in/parshva-shah-b40683193/
GitHub      : github.com/parshvashahh

Description:
    A full-stack quantitative research platform combining:

    1. SQL Analytics Pipeline
       - SQLite database with 5 normalized tables
       - 8 advanced SQL queries using window functions,
         CTEs, joins, aggregations, and subqueries
       - Sector analysis, drawdown, regime detection

    2. Time Series Forecasting Engine
       - ADF stationarity testing
       - ACF/PACF autocorrelation analysis
       - ARIMA(2,1,0) with OLS estimation
       - EWMA volatility (RiskMetrics lambda=0.94)
       - Rolling walk-forward forecast evaluation
       - Exponential smoothing

    3. Market Risk Dashboard
       - Cross-asset correlation heatmap
       - Macro factor correlation analysis
       - Volatility regime classification
       - Sector cumulative return comparison
       - ARIMA forecast visualization
       - Rolling EWMA volatility

Key Concepts Demonstrated:
    - Time series stationarity (ADF test)
    - ARIMA modeling and forecasting
    - EWMA volatility estimation
    - SQL window functions (ROW_NUMBER, RANK, LAG)
    - SQL CTEs and complex joins
    - Macro factor analysis
    - Volatility regime detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from sql_analytics import run_sql_analytics
from time_series import (run_time_series_analysis, compute_acf,
                          ewma_volatility, SimpleARIMA)

DB_PATH = 'financial_market.db'

TARGET_STOCKS = ['TCS', 'HDFCBANK', 'RELIANCE', 'HINDUNILVR', 'SUNPHARMA']
SECTORS       = ['IT', 'Financials', 'Energy', 'FMCG', 'Healthcare']

#DATA LOADING
def load_data():
    """Load all data from SQLite and CSV files."""
    conn = sqlite3.connect(DB_PATH)

    # Load prices — pivot from long to wide
    prices_long = pd.read_sql(
        "SELECT date, symbol, close_price FROM stock_prices",
        conn, parse_dates=['date'])
    price_df = prices_long.pivot(
        index='date', columns='symbol', values='close_price')
    price_df.index.name = 'date'

    # Load returns
    returns_long = pd.read_sql(
        "SELECT date, symbol, log_return FROM stock_returns",
        conn, parse_dates=['date'])
    returns_df = returns_long.pivot(
        index='date', columns='symbol', values='log_return')
    returns_df.index.name = 'date'

    # Load macro
    macro_df = pd.read_sql(
        "SELECT * FROM macro_indicators",
        conn, parse_dates=['date'], index_col='date')

    conn.close()
    print(f"Data loaded: {len(price_df)} days | "
          f"{len(price_df.columns)} stocks")
    return price_df, returns_df, macro_df

#MACRO CORRELATION ANALYSIS
def macro_correlation_analysis(returns_df, macro_df):
    """Compute correlation between stock returns and macro indicators."""
    # Align dates
    common_dates = returns_df.index.intersection(macro_df.index)
    ret_aligned   = returns_df.loc[common_dates, TARGET_STOCKS]
    macro_aligned = macro_df.loc[common_dates]

    results = {}
    for stock in TARGET_STOCKS:
        corr_row = {}
        for macro_var in macro_aligned.columns:
            corr = ret_aligned[stock].corr(macro_aligned[macro_var])
            corr_row[macro_var] = round(corr, 4)
        results[stock] = corr_row

    return pd.DataFrame(results).T

#VOLATILITY REGIME SUMMARY
def volatility_regime_summary(sql_results):
    """Summarize volatility regime distribution."""
    regime_df = sql_results['vol_regime']
    summary = regime_df.groupby(
        ['symbol', 'vol_regime']).size().unstack(fill_value=0)
    summary['Total'] = summary.sum(axis=1)
    for col in ['NORMAL', 'ELEVATED', 'HIGH_VOL']:
        if col in summary.columns:
            summary[f'{col}_pct'] = (
                summary[col] / summary['Total'] * 100).round(1)
    return summary

#SECTOR PERFORMANCE SUMMARY

def sector_performance_summary(sql_results, returns_df):
    """Compute annualized return and Sharpe by sector."""
    conn = sqlite3.connect(DB_PATH)
    metadata = pd.read_sql("SELECT * FROM stock_metadata", conn)
    conn.close()

    sector_map = dict(zip(metadata['symbol'], metadata['sector']))
    results = []

    for sector in SECTORS:
        sector_stocks = [s for s, sec in sector_map.items()
                        if sec == sector and s in returns_df.columns]
        if not sector_stocks:
            continue

        sector_returns = returns_df[sector_stocks].mean(axis=1)
        ann_return = sector_returns.mean() * 252 * 100
        ann_vol    = sector_returns.std() * np.sqrt(252) * 100
        sharpe     = (ann_return - 6.5) / ann_vol if ann_vol > 0 else 0
        total_ret  = (np.exp(sector_returns.sum()) - 1) * 100

        results.append({
            'Sector'         : sector,
            'Ann Return %'   : round(ann_return, 2),
            'Ann Volatility %': round(ann_vol, 2),
            'Sharpe Ratio'   : round(sharpe, 3),
            'Total Return %' : round(total_ret, 2),
        })

    return pd.DataFrame(results)

#VISUALIZATION MAIN DASHBOARD
def plot_research_dashboard(price_df, returns_df, macro_df,
                             sql_results, ts_results, macro_corr):
    """Create comprehensive 6-panel research dashboard."""
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Quantitative Research Engine — Market Analytics Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.38)

    #Panel 1 Sector Cumulative Returns
    ax1 = fig.add_subplot(gs[0, :2])
    conn = sqlite3.connect(DB_PATH)
    meta = pd.read_sql("SELECT * FROM stock_metadata", conn)
    conn.close()
    sector_map = dict(zip(meta['symbol'], meta['sector']))
    colors_s = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    for i, sector in enumerate(SECTORS):
        stocks = [s for s, sec in sector_map.items()
                 if sec == sector and s in returns_df.columns]
        if stocks:
            sector_ret = returns_df[stocks].mean(axis=1)
            cum_ret    = (np.exp(sector_ret.cumsum()) - 1) * 100
            ax1.plot(cum_ret.index, cum_ret.values,
                    color=colors_s[i], linewidth=1.5,
                    label=f'{sector}')

    ax1.set_title('Sector Cumulative Returns (%)', fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    #Panel 2 ARIMA Forecast for TCS
    ax2 = fig.add_subplot(gs[0, 2])
    stock = 'TCS'
    if stock in ts_results:
        prices = price_df[stock].dropna()
        model  = SimpleARIMA(p=2, d=1, q=0)
        model.fit(prices.values)
        fc     = model.forecast(prices.values, steps=30)

        last_n = 60
        ax2.plot(range(last_n), prices.values[-last_n:],
                color='steelblue', linewidth=1.5, label='Historical')
        ax2.plot(range(last_n, last_n + 30), fc,
                color='red', linewidth=1.5, linestyle='--',
                label='ARIMA Forecast')
        ax2.axvline(last_n, color='black',
                   linestyle=':', linewidth=1)
        ax2.set_title(f'{stock} — ARIMA(2,1,0) Forecast',
                     fontweight='bold')
        ax2.set_ylabel('Price (INR)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    #Panel 3 EWMA Volatility
    ax3 = fig.add_subplot(gs[1, :2])
    vol_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    for i, stock in enumerate(TARGET_STOCKS):
        if stock in returns_df.columns:
            ewma_vol = ewma_volatility(returns_df[stock])
            ax3.plot(returns_df.index[-500:],
                    ewma_vol[-500:],
                    color=vol_colors[i], linewidth=1.0,
                    alpha=0.8, label=stock)

    ax3.set_title('EWMA Volatility — RiskMetrics (lambda=0.94)',
                 fontweight='bold')
    ax3.set_ylabel('Annualised Volatility (%)')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Panel 4 Macro Correlation Heatmap
    ax4 = fig.add_subplot(gs[1, 2])
    macro_labels = ['repo_rate', 'cpi', 'usd_inr',
                   'crude_oil', 'nifty_vix']
    corr_data = macro_corr[macro_labels].values
    im = ax4.imshow(corr_data, cmap='RdYlGn',
                   vmin=-0.3, vmax=0.3, aspect='auto')
    ax4.set_xticks(range(len(macro_labels)))
    ax4.set_yticks(range(len(TARGET_STOCKS)))
    ax4.set_xticklabels(['Repo', 'CPI', 'USD/INR',
                        'Crude', 'VIX'],
                       rotation=30, fontsize=7)
    ax4.set_yticklabels(TARGET_STOCKS, fontsize=7)
    plt.colorbar(im, ax=ax4)
    ax4.set_title('Stock-Macro Correlation', fontweight='bold')
    for i in range(len(TARGET_STOCKS)):
        for j in range(len(macro_labels)):
            ax4.text(j, i, f'{corr_data[i,j]:.2f}',
                    ha='center', va='center', fontsize=6)

    #Panel 5 Cross Asset Correlation
    ax5 = fig.add_subplot(gs[2, :2])
    corr_matrix = returns_df[TARGET_STOCKS].corr()
    im2 = ax5.imshow(corr_matrix.values, cmap='RdYlGn',
                    vmin=0, vmax=1, aspect='auto')
    ax5.set_xticks(range(len(TARGET_STOCKS)))
    ax5.set_yticks(range(len(TARGET_STOCKS)))
    ax5.set_xticklabels(TARGET_STOCKS, rotation=30, fontsize=8)
    ax5.set_yticklabels(TARGET_STOCKS, fontsize=8)
    plt.colorbar(im2, ax=ax5)
    ax5.set_title('Cross-Asset Return Correlation', fontweight='bold')
    for i in range(len(TARGET_STOCKS)):
        for j in range(len(TARGET_STOCKS)):
            ax5.text(j, i, f'{corr_matrix.values[i,j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    fontweight='bold')

    #Panel 6 Volatility Regime Distribution
    ax6 = fig.add_subplot(gs[2, 2])
    regime_df = sql_results['vol_regime']
    regime_counts = regime_df.groupby('vol_regime').size()
    regime_pct    = regime_counts / regime_counts.sum() * 100
    colors_r = {'NORMAL': '#4CAF50', 'ELEVATED': '#FF9800',
                'HIGH_VOL': '#F44336'}
    bars = ax6.bar(regime_pct.index, regime_pct.values,
                  color=[colors_r.get(r, 'gray')
                        for r in regime_pct.index],
                  alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, regime_pct.values):
        ax6.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9)
    ax6.set_title('Volatility Regime Distribution\n(All Stocks)',
                 fontweight='bold')
    ax6.set_ylabel('Percentage of Days (%)')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.savefig('research_dashboard.png', dpi=150,
               bbox_inches='tight')
    plt.close()
    print("Dashboard saved: research_dashboard.png")

#MAIN REPORT
def print_research_report(ts_results, sector_perf,
                           macro_corr, sql_results):
    """Print comprehensive research report."""
    print("\n" + "="*65)
    print("   QUANTITATIVE RESEARCH ENGINE — ANALYSIS REPORT")
    print("="*65)

    print("\n── Time Series Analysis Results ─────────────────────────")
    print(f"{'Stock':<12} {'ADF Prices':<16} {'ADF Returns':<16} "
          f"{'RMSE':>8} {'Dir Acc':>9}")
    print("-"*65)
    for stock, res in ts_results.items():
        print(f"{stock:<12} "
              f"{res['adf_prices']['conclusion']:<16} "
              f"{res['adf_returns']['conclusion']:<16} "
              f"{res['eval_metrics']['RMSE']:>8.2f} "
              f"{res['eval_metrics']['Dir_Accuracy']:>8.1f}%")

    print("\n── Sector Performance Summary ───────────────────────────")
    print(sector_perf.to_string(index=False))

    print("\n── Macro Correlation Analysis ───────────────────────────")
    print("  (Correlation between daily returns and macro indicators)")
    print(macro_corr.round(3).to_string())

    print("\n── SQL Analytics Summary ────────────────────────────────")
    regime_df   = sql_results['vol_regime']
    regime_dist = regime_df.groupby('vol_regime').size()
    total       = regime_dist.sum()
    for regime, count in regime_dist.items():
        print(f"  {regime:<12}: {count:,} days "
              f"({count/total*100:.1f}%)")

    monthly_df = sql_results['monthly']
    best_month = monthly_df.loc[
        monthly_df['monthly_return_pct'].idxmax()]
    worst_month = monthly_df.loc[
        monthly_df['monthly_return_pct'].idxmin()]
    print(f"\n  Best Month  : {best_month['symbol']} "
          f"{best_month['year_month']} "
          f"(+{best_month['monthly_return_pct']:.2f}%)")
    print(f"  Worst Month : {worst_month['symbol']} "
          f"{worst_month['year_month']} "
          f"({worst_month['monthly_return_pct']:.2f}%)")

    print("\n" + "="*65)
    
#EXPORT RESULTS
def export_results(ts_results, sector_perf,
                   macro_corr, sql_results):
    """Export all results to CSV."""
    sector_perf.to_csv('sector_performance.csv', index=False)
    macro_corr.to_csv('macro_correlation.csv')
    sql_results['monthly'].to_csv('monthly_returns.csv', index=False)
    sql_results['drawdown'].to_csv('drawdown_analysis.csv', index=False)

    #Time series summary
    ts_summary = []
    for stock, res in ts_results.items():
        ts_summary.append({
            'Stock'          : stock,
            'ADF_Price'      : res['adf_prices']['conclusion'],
            'ADF_Returns'    : res['adf_returns']['conclusion'],
            'RMSE'           : res['eval_metrics']['RMSE'],
            'MAE'            : res['eval_metrics']['MAE'],
            'Dir_Accuracy_%' : res['eval_metrics']['Dir_Accuracy'],
            'Last_Price'     : round(res['last_price'], 2),
            'ARIMA_Forecast_10D': round(res['arima_forecast'][-1], 2),
        })
    pd.DataFrame(ts_summary).to_csv('ts_forecast_results.csv',
                                     index=False)
    print("Results exported to CSV files")

#RUN ENGINE
if __name__ == '__main__':
    print("=" * 65)
    print("   QUANTITATIVE RESEARCH ENGINE — STARTING")
    print("=" * 65)

    #Load data
    print("\nLoading data from SQLite database...")
    price_df, returns_df, macro_df = load_data()

    #Module 1: SQL Analytics
    print("\n[MODULE 1] SQL ANALYTICS PIPELINE")
    sql_results = run_sql_analytics()

    # Module 2: Time Series
    print("\n[MODULE 2] TIME SERIES FORECASTING")
    ts_results = run_time_series_analysis(
        price_df, returns_df, TARGET_STOCKS)

    #Module 3: Additional Analysis
    print("\n[MODULE 3] MACRO & SECTOR ANALYSIS")
    macro_corr  = macro_correlation_analysis(returns_df, macro_df)
    sector_perf = sector_performance_summary(sql_results, returns_df)
    print("Macro correlation and sector analysis complete")

    # Report
    print_research_report(ts_results, sector_perf,
                           macro_corr, sql_results)

    #Dashboard
    print("\nGenerating research dashboard...")
    plot_research_dashboard(price_df, returns_df, macro_df,
                            sql_results, ts_results, macro_corr)

    #Export
    export_results(ts_results, sector_perf,
                   macro_corr, sql_results)

    print("\n" + "="*65)
    print("   PROJECT 6 COMPLETE")
    print("="*65)
