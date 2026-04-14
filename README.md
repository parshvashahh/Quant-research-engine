# Quantitative Research Engine Time Series + SQL + Risk Analytics

A full-stack **quantitative research platform** combining SQL analytics pipeline, time series forecasting, and market risk dashboard — built specifically for Quantitative Research Analyst roles.

---

#What This Project Does

#Module 1 — SQL Analytics Pipeline
- SQLite database with **5 normalized tables** (stock_prices, stock_returns, macro_indicators, stock_volume, stock_metadata)
- **8 advanced SQL queries** using Window Functions, CTEs, Joins, Aggregations
- Sector performance analysis, drawdown computation, volatility regime detection

#Module 2 — Time Series Forecasting Engine
- **ADF Stationarity Test** — validates whether series is stationary before modeling
- **ACF/PACF Analysis** — autocorrelation structure for lag selection
- **ARIMA(2,1,0)** — fitted via OLS with AIC/BIC model selection
- **EWMA Volatility** — RiskMetrics lambda=0.94 time-varying volatility
- **Rolling Walk-Forward Evaluation** — RMSE, MAE, MAPE, Directional Accuracy
- **Exponential Smoothing** — simple baseline forecasting

# Module 3 — Market Risk Dashboard
- Cross-asset correlation heatmap
- Macro factor correlation (Repo Rate, CPI, USD/INR, Crude Oil, VIX)
- Sector cumulative return comparison
- Volatility regime classification (Normal / Elevated / High Vol)
- ARIMA forecast visualization

---

#Project Structure

```
project6_quant_research_engine/
├── quant_research_engine.py    # Main orchestrator — run this
├── sql_analytics.py            # Module 1: All SQL queries
├── time_series.py              # Module 2: ARIMA, EWMA, ADF
├── generate_data.py            # Dataset generator
├── financial_market.db         # SQLite database (5 tables)
├── stock_prices.csv            # 15 stocks x 1261 days
├── stock_returns.csv           # Log returns dataset
├── macro_indicators.csv        # Repo rate, CPI, USD/INR, Crude, VIX
├── research_dashboard.png      # Output: 6-panel dashboard
├── sector_performance.csv      # Output: sector analytics
├── ts_forecast_results.csv     # Output: ARIMA forecast results
├── monthly_returns.csv         # Output: monthly return summary
├── drawdown_analysis.csv       # Output: drawdown per stock
├── requirements.txt
└── README.md
```

---



---

#SQL Queries — Key Techniques

| Query | SQL Features Used |
|-------|-------------------|
| Rolling 30-Day Volatility | Window function, ROWS BETWEEN |
| Sector Average Returns | GROUP BY, AVG, JOIN |
| Top 5 Extreme Days | RANK() OVER PARTITION BY |
| Cumulative Returns by Sector | SUM() OVER with ORDER BY |
| Volatility Regime Detection | CTE, CASE WHEN, subquery |
| Macro Factor Join | Multi-table JOIN, date alignment |
| Monthly Performance | STRFTIME, aggregation |
| Drawdown Analysis | MAX() OVER, arithmetic in SELECT |

---

# Key Output Numbers

| Metric | Value |
|--------|-------|
| Financials Sharpe | 1.136 — best sector |
| IT Total Return | 135.98% over 5 years |
| Volatility Regime — Normal | 57.7% of days |
| Volatility Regime — High Vol | 11.2% of days |
| All returns ADF test | Stationary (as expected) |
| Directional Accuracy (TCS) | 53.0% — above random |

---



# Tech Stack
`Python` `SQL` `SQLite` `Pandas` `NumPy` `SciPy` `Matplotlib`

---

