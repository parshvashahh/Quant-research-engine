"""
Generate realistic financial market dataset for Project 6
Simulates NIFTY 50 style equity data with macroeconomic indicators
Saves to SQLite database AND CSV files
by parshva shah, Github: parshvashahh
"""

import numpy as np
import pandas as pd
import sqlite3
import os

np.random.seed(42)

#Parameters
N_DAYS    = 1260   # 5 years of trading data
N_STOCKS  = 15     # 15 stocks across 5 sectors
START     = '2019-01-01'

#Stock Universe
STOCKS = {
    # IT Sector
    'TCS'       : {'sector': 'IT',         'mu': 0.18, 'sigma': 0.20, 'start_price': 2200},
    'INFY'      : {'sector': 'IT',         'mu': 0.20, 'sigma': 0.22, 'start_price': 900},
    'WIPRO'     : {'sector': 'IT',         'mu': 0.15, 'sigma': 0.24, 'start_price': 400},
    # Financials
    'HDFCBANK'  : {'sector': 'Financials', 'mu': 0.16, 'sigma': 0.24, 'start_price': 1200},
    'ICICIBANK' : {'sector': 'Financials', 'mu': 0.17, 'sigma': 0.26, 'start_price': 500},
    'KOTAKBANK' : {'sector': 'Financials', 'mu': 0.16, 'sigma': 0.22, 'start_price': 1500},
    # Energy
    'RELIANCE'  : {'sector': 'Energy',     'mu': 0.14, 'sigma': 0.22, 'start_price': 1500},
    'ONGC'      : {'sector': 'Energy',     'mu': 0.10, 'sigma': 0.28, 'start_price': 120},
    'BPCL'      : {'sector': 'Energy',     'mu': 0.11, 'sigma': 0.26, 'start_price': 350},
    # FMCG
    'HINDUNILVR': {'sector': 'FMCG',       'mu': 0.12, 'sigma': 0.18, 'start_price': 2000},
    'NESTLEIND' : {'sector': 'FMCG',       'mu': 0.13, 'sigma': 0.16, 'start_price': 15000},
    'BRITANNIA' : {'sector': 'FMCG',       'mu': 0.11, 'sigma': 0.19, 'start_price': 3500},
    # Healthcare
    'SUNPHARMA' : {'sector': 'Healthcare', 'mu': 0.15, 'sigma': 0.24, 'start_price': 600},
    'DRREDDY'   : {'sector': 'Healthcare', 'mu': 0.14, 'sigma': 0.22, 'start_price': 3500},
    'CIPLA'     : {'sector': 'Healthcare', 'mu': 0.13, 'sigma': 0.23, 'start_price': 700},
}

symbols = list(STOCKS.keys())
n_stocks = len(symbols)

#Correlation Matrix
# Sectors: IT(3), Financials(3), Energy(3), FMCG(3), Healthcare(3)
def build_corr_matrix():
    corr = np.eye(n_stocks)
    for i in range(n_stocks):
        for j in range(n_stocks):
            if i == j:
                corr[i,j] = 1.0
            else:
                si = STOCKS[symbols[i]]['sector']
                sj = STOCKS[symbols[j]]['sector']
                if si == sj:
                    corr[i,j] = np.random.uniform(0.60, 0.80)
                else:
                    corr[i,j] = np.random.uniform(0.25, 0.50)
    # Make symmetric positive definite
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    # Ensure positive definite
    min_eig = np.linalg.eigvalsh(corr).min()
    if min_eig < 0:
        corr += (-min_eig + 0.01) * np.eye(n_stocks)
        d = np.diag(1.0 / np.sqrt(np.diag(corr)))
        corr = d @ corr @ d
    return corr

corr_matrix = build_corr_matrix()

#Simulate Stock Prices
dt = 1/252
sigmas = np.array([STOCKS[s]['sigma'] for s in symbols])
mus    = np.array([STOCKS[s]['mu']    for s in symbols])
starts = np.array([STOCKS[s]['start_price'] for s in symbols])

cov_daily = np.outer(sigmas, sigmas) * corr_matrix / 252
chol      = np.linalg.cholesky(cov_daily)

prices = np.zeros((N_DAYS + 1, n_stocks))
prices[0] = starts

for t in range(1, N_DAYS + 1):
    z      = np.random.standard_normal(n_stocks)
    cz     = chol @ z
    drift  = (mus - 0.5 * sigmas**2) * dt
    prices[t] = prices[t-1] * np.exp(drift + cz)

dates = pd.bdate_range(start=START, periods=N_DAYS + 1)

#Price DataFrame
price_df = pd.DataFrame(prices, index=dates, columns=symbols)
price_df.index.name = 'date'
price_df = price_df.round(2)

# Returns DataFrame
returns_df = np.log(price_df / price_df.shift(1)).dropna()
returns_df = returns_df.round(6)

# Macro Indicators
#Simulate: repo rate, CPI, GDP growth, USD/INR, crude oil
macro_dates = pd.date_range(start=START, periods=N_DAYS+1, freq='D')
macro_dates = macro_dates[macro_dates.isin(dates)]

repo_rate   = 6.0 + np.cumsum(np.random.normal(0, 0.02, len(macro_dates)))
repo_rate   = np.clip(repo_rate, 4.0, 8.0)
cpi         = 5.0 + np.cumsum(np.random.normal(0, 0.03, len(macro_dates)))
cpi         = np.clip(cpi, 2.0, 9.0)
usd_inr     = 72.0 + np.cumsum(np.random.normal(0.01, 0.15, len(macro_dates)))
usd_inr     = np.clip(usd_inr, 68.0, 88.0)
crude_oil   = 60.0 + np.cumsum(np.random.normal(0, 0.5, len(macro_dates)))
crude_oil   = np.clip(crude_oil, 30.0, 120.0)
nifty_vix   = 15.0 + np.abs(np.random.normal(0, 2, len(macro_dates)))
nifty_vix   = np.clip(nifty_vix, 10.0, 50.0)

macro_df = pd.DataFrame({
    'date'      : macro_dates,
    'repo_rate' : repo_rate.round(2),
    'cpi'       : cpi.round(2),
    'usd_inr'   : usd_inr.round(2),
    'crude_oil' : crude_oil.round(2),
    'nifty_vix' : nifty_vix.round(2),
})
macro_df.set_index('date', inplace=True)

#Volume Data
volume_data = {}
for sym in symbols:
    base_vol = np.random.randint(500000, 5000000)
    volume_data[sym] = (base_vol + np.random.randint(-base_vol//2,
                        base_vol//2, N_DAYS+1)).clip(100000)

volume_df = pd.DataFrame(volume_data, index=dates)
volume_df.index.name = 'date'

#Save to SQLite Database 
db_path = 'financial_market.db'
conn = sqlite3.connect(db_path)

#Table 1: stock_prices
price_long = price_df.reset_index().melt(
    id_vars='date', var_name='symbol', value_name='close_price')
price_long['sector'] = price_long['symbol'].map(
    {s: STOCKS[s]['sector'] for s in symbols})
price_long.to_sql('stock_prices', conn, if_exists='replace', index=False)

#Table 2: stock_returns
returns_long = returns_df.reset_index().melt(
    id_vars='date', var_name='symbol', value_name='log_return')
returns_long.to_sql('stock_returns', conn, if_exists='replace', index=False)

#Table 3: macro_indicators
macro_df.reset_index().to_sql('macro_indicators', conn,
                               if_exists='replace', index=False)

#Table 4: stock_volume
volume_long = volume_df.reset_index().melt(
    id_vars='date', var_name='symbol', value_name='volume')
volume_long.to_sql('stock_volume', conn, if_exists='replace', index=False)

#Table 5: stock_metadata
metadata = pd.DataFrame([
    {'symbol': s, 'sector': STOCKS[s]['sector'],
     'start_price': STOCKS[s]['start_price']}
    for s in symbols])
metadata.to_sql('stock_metadata', conn, if_exists='replace', index=False)

conn.close()

#Save CSVs
price_df.to_csv('stock_prices.csv')
returns_df.to_csv('stock_returns.csv')
macro_df.to_csv('macro_indicators.csv')

print(f"Database created: {db_path}")
print(f"Tables: stock_prices, stock_returns, macro_indicators, stock_volume, stock_metadata")
print(f"Stocks: {n_stocks} | Days: {N_DAYS} | Date range: {dates[0].date()} to {dates[-1].date()}")
print(f"CSV files saved: stock_prices.csv, stock_returns.csv, macro_indicators.csv")
