"""SQL Analytics Engine
Runs all quantitative analytics queries against
the SQLite database — computing rolling metrics,
sector analysis, correlation, and risk indicators
entirely in SQL. by parshva, github: parshvashahh
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'financial_market.db'


def get_conn():
    return sqlite3.connect(DB_PATH)


#1 QUERY Rolling 30 Day Volatility
QUERY_ROLLING_VOL = """
SELECT
    date,
    symbol,
    log_return,
    AVG(log_return * log_return) OVER (
        PARTITION BY symbol
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS variance_30d,
    COUNT(*) OVER (
        PARTITION BY symbol
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS window_size
FROM stock_returns
ORDER BY symbol, date
"""

#2 QUERY Sector Average Returns
QUERY_SECTOR_RETURNS = """
SELECT
    r.date,
    m.sector,
    AVG(r.log_return)  AS avg_sector_return,
    COUNT(r.symbol)    AS stock_count,
    MIN(r.log_return)  AS min_return,
    MAX(r.log_return)  AS max_return
FROM stock_returns r
JOIN stock_metadata m ON r.symbol = m.symbol
GROUP BY r.date, m.sector
ORDER BY r.date, m.sector
"""

#3 QUERY Top 10 Best and Worst Days Per Stock
QUERY_EXTREME_DAYS = """
SELECT
    symbol,
    date,
    log_return,
    CASE WHEN log_return > 0 THEN 'BEST' ELSE 'WORST' END AS day_type,
    RANK() OVER (
        PARTITION BY symbol,
        CASE WHEN log_return > 0 THEN 'BEST' ELSE 'WORST' END
        ORDER BY ABS(log_return) DESC
    ) AS rank_num
FROM stock_returns
"""

#4 QUERY Cumulative Returns by Sector
QUERY_CUMULATIVE = """
SELECT
    r.date,
    m.sector,
    EXP(SUM(AVG(r.log_return)) OVER (
        PARTITION BY m.sector
        ORDER BY r.date
    )) - 1 AS cumulative_return
FROM stock_returns r
JOIN stock_metadata m ON r.symbol = m.symbol
GROUP BY r.date, m.sector
ORDER BY m.sector, r.date
"""

#5 QUERY Volatility Regime Detection
QUERY_VOL_REGIME = """
WITH daily_vol AS (
    SELECT
        date,
        symbol,
        ABS(log_return) AS abs_return
    FROM stock_returns
),
vol_stats AS (
    SELECT
        symbol,
        AVG(abs_return)           AS mean_vol,
        AVG(abs_return * abs_return) - AVG(abs_return)*AVG(abs_return)
                                  AS var_vol
    FROM daily_vol
    GROUP BY symbol
)
SELECT
    d.date,
    d.symbol,
    d.abs_return,
    v.mean_vol,
    CASE
        WHEN d.abs_return > v.mean_vol * 2 THEN 'HIGH_VOL'
        WHEN d.abs_return > v.mean_vol     THEN 'ELEVATED'
        ELSE                                    'NORMAL'
    END AS vol_regime
FROM daily_vol d
JOIN vol_stats v ON d.symbol = v.symbol
ORDER BY d.symbol, d.date
"""

#6 QUERY Macro Correlation Analysis 
QUERY_MACRO_JOIN = """
SELECT
    r.date,
    r.symbol,
    r.log_return,
    m.repo_rate,
    m.cpi,
    m.usd_inr,
    m.crude_oil,
    m.nifty_vix
FROM stock_returns r
JOIN macro_indicators m ON r.date = m.date
ORDER BY r.symbol, r.date
"""

#7 QUERY Monthly Performance Summary
QUERY_MONTHLY = """
SELECT
    symbol,
    STRFTIME('%Y-%m', date) AS year_month,
    SUM(log_return)         AS monthly_log_return,
    COUNT(*)                AS trading_days,
    MIN(log_return)         AS worst_day,
    MAX(log_return)         AS best_day,
    AVG(log_return * log_return) AS avg_variance
FROM stock_returns
GROUP BY symbol, STRFTIME('%Y-%m', date)
ORDER BY symbol, year_month
"""

#8 QUERY Drawdown Analysis via SQL
QUERY_DRAWDOWN = """
WITH price_data AS (
    SELECT
        date,
        symbol,
        close_price,
        MAX(close_price) OVER (
            PARTITION BY symbol
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_max
    FROM stock_prices
)
SELECT
    date,
    symbol,
    close_price,
    running_max,
    (close_price - running_max) / running_max AS drawdown_pct
FROM price_data
ORDER BY symbol, date
"""


def run_sql_analytics():
    """Run all SQL queries and return results."""
    conn = get_conn()
    results = {}

    print("Running SQL Analytics Queries...")
    print("-" * 50)

    # Query 1: Rolling Volatility
    df = pd.read_sql(QUERY_ROLLING_VOL, conn)
    df['rolling_vol_ann'] = np.sqrt(df['variance_30d'] * 252) * 100
    results['rolling_vol'] = df
    print(f"Query 1 Rolling Volatility    : {len(df):,} rows")

    # Query 2: Sector Returns
    df2 = pd.read_sql(QUERY_SECTOR_RETURNS, conn)
    results['sector_returns'] = df2
    print(f"Query 2 Sector Returns        : {len(df2):,} rows")

    # Query 3: Extreme Days
    df3 = pd.read_sql(QUERY_EXTREME_DAYS, conn)
    df3 = df3[df3['rank_num'] <= 5]
    results['extreme_days'] = df3
    print(f"Query 3 Extreme Days          : {len(df3):,} rows")

    # Query 4: Cumulative Returns
    df4 = pd.read_sql(QUERY_CUMULATIVE, conn)
    results['cumulative'] = df4
    print(f"Query 4 Cumulative Returns    : {len(df4):,} rows")

    # Query 5: Vol Regime
    df5 = pd.read_sql(QUERY_VOL_REGIME, conn)
    results['vol_regime'] = df5
    print(f"Query 5 Volatility Regime     : {len(df5):,} rows")

    # Query 6: Macro Join
    df6 = pd.read_sql(QUERY_MACRO_JOIN, conn)
    results['macro_join'] = df6
    print(f"Query 6 Macro Correlation     : {len(df6):,} rows")

    # Query 7: Monthly Summary
    df7 = pd.read_sql(QUERY_MONTHLY, conn)
    df7['monthly_return_pct'] = (np.exp(df7['monthly_log_return']) - 1) * 100
    results['monthly'] = df7
    print(f"Query 7 Monthly Summary       : {len(df7):,} rows")

    # Query 8: Drawdown
    df8 = pd.read_sql(QUERY_DRAWDOWN, conn)
    results['drawdown'] = df8
    print(f"Query 8 Drawdown Analysis     : {len(df8):,} rows")

    conn.close()
    print("-" * 50)
    print("All SQL queries completed successfully")
    return results
