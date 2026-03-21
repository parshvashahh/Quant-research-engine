"""parshva shah, github: parshvashahh
Time Series Forecasting Engine
ARIMA-style forecasting using statistical methods
compatible with standard scipy/numpy stack.
Includes stationarity testing, ACF/PACF analysis,
rolling forecast, and forecast accuracy metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


#ADF Stationarity Test
def adf_test_simple(series, max_lag=10):
    """
    Simplified Augmented Dickey-Fuller test.
    Tests whether a time series is stationary.
    H0: Series has unit root (non-stationary)
    H1: Series is stationary
    """
    series = series.dropna().values
    n = len(series)

    # Compute first differences
    dy = np.diff(series)
    y_lag = series[:-1]

    # Simple regression: dy = alpha + beta*y_lag + epsilon
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        y_hat = X @ beta
        residuals = dy - y_hat
        sigma2 = np.var(residuals)
        se_beta = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
        t_stat = beta[1] / se_beta if se_beta > 0 else 0
    except:
        t_stat = 0

    # Approximate p-value using normal distribution
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    is_stationary = p_value < 0.05

    return {
        'test_statistic': round(t_stat, 4),
        'p_value': round(p_value, 4),
        'is_stationary': is_stationary,
        'conclusion': 'Stationary' if is_stationary else 'Non-Stationary'
    }


#ACF Computation
def compute_acf(series, max_lags=20):
    """Compute Autocorrelation Function."""
    series = series.dropna().values
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)

    acf_values = []
    for lag in range(max_lags + 1):
        if var == 0:
            acf_values.append(0)
        else:
            cov = np.mean((series[:n-lag] - mean) * (series[lag:] - mean))
            acf_values.append(cov / var)

    conf_interval = 1.96 / np.sqrt(n)
    return np.array(acf_values), conf_interval


#ARIMA Manual Implementation
class SimpleARIMA:
    """
    Manual ARIMA(p,d,q) implementation using OLS.
    p = autoregressive order
    d = differencing order
    q = moving average order (simplified)
    """

    def __init__(self, p=1, d=1, q=0):
        self.p = p
        self.d = d
        self.q = q
        self.coefficients = None
        self.fitted_values = None
        self.residuals = None
        self.aic = None
        self.bic = None

    def difference(self, series, d):
        """Apply d-order differencing."""
        result = series.copy()
        for _ in range(d):
            result = np.diff(result)
        return result

    def fit(self, series):
        """Fit ARIMA model using OLS regression."""
        series = np.array(series)

        # Step 1: Differencing
        if self.d > 0:
            diff_series = self.difference(series, self.d)
        else:
            diff_series = series.copy()

        n = len(diff_series)
        if n <= self.p + 5:
            return self

        # Step 2: Build lagged feature matrix
        y = diff_series[self.p:]
        X_cols = [np.ones(len(y))]
        for lag in range(1, self.p + 1):
            X_cols.append(diff_series[self.p - lag: n - lag])

        X = np.column_stack(X_cols)

        # Step 3: OLS estimation
        try:
            self.coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
            y_hat = X @ self.coefficients
            self.residuals = y - y_hat
            self.fitted_values = y_hat

            # Information criteria
            n_obs = len(y)
            k = len(self.coefficients)
            sigma2 = np.var(self.residuals)
            if sigma2 > 0:
                log_lik = -0.5 * n_obs * (np.log(2 * np.pi * sigma2) + 1)
                self.aic = -2 * log_lik + 2 * k
                self.bic = -2 * log_lik + k * np.log(n_obs)
        except:
            self.coefficients = np.array([0.0, 0.1])

        return self

    def forecast(self, series, steps=10):
        """Generate out-of-sample forecasts."""
        if self.coefficients is None:
            return np.zeros(steps)

        series = np.array(series)
        if self.d > 0:
            diff_series = self.difference(series, self.d)
        else:
            diff_series = series.copy()

        forecasts = []
        history = list(diff_series)

        for _ in range(steps):
            lags = [history[-lag] for lag in range(1, self.p + 1)]
            x = np.array([1.0] + lags)
            if len(x) == len(self.coefficients):
                pred_diff = x @ self.coefficients
            else:
                pred_diff = history[-1] * 0.1

            forecasts.append(pred_diff)
            history.append(pred_diff)

        # Undifference
        if self.d > 0:
            last_value = series[-1]
            undiff = [last_value]
            for f in forecasts:
                undiff.append(undiff[-1] + f)
            return np.array(undiff[1:])

        return np.array(forecasts)


#Rolling Forecast & Evaluation
def rolling_forecast_evaluation(series, train_size=0.8,
                                  forecast_horizon=5):
    """
    Walk-forward rolling forecast evaluation.
    Trains on expanding window, forecasts h steps ahead.
    Computes RMSE, MAE, MAPE, directional accuracy.
    """
    series = series.dropna().values
    n = len(series)
    train_end = int(n * train_size)

    actuals = []
    predictions = []

    for i in range(train_end, n - forecast_horizon, forecast_horizon):
        train = series[:i]
        actual = series[i:i + forecast_horizon]

        model = SimpleARIMA(p=2, d=1, q=0)
        model.fit(train)
        pred = model.forecast(train, steps=forecast_horizon)

        if len(pred) == len(actual):
            actuals.extend(actual)
            predictions.extend(pred)

    actuals = np.array(actuals)
    predictions = np.array(predictions)

    if len(actuals) == 0:
        return {'RMSE': 0, 'MAE': 0, 'MAPE': 0, 'Dir_Accuracy': 0}

    rmse = np.sqrt(np.mean((actuals - predictions)**2))
    mae  = np.mean(np.abs(actuals - predictions))
    mape = np.mean(np.abs((actuals - predictions) /
                           np.clip(np.abs(actuals), 1e-8, None))) * 100

    # Directional accuracy
    if len(actuals) > 1:
        dir_actual = np.sign(np.diff(actuals))
        dir_pred   = np.sign(np.diff(predictions))
        dir_acc    = np.mean(dir_actual == dir_pred) * 100
    else:
        dir_acc = 50.0

    return {
        'RMSE': round(rmse, 4),
        'MAE' : round(mae, 4),
        'MAPE': round(mape, 2),
        'Dir_Accuracy': round(dir_acc, 1),
        'N_forecasts': len(actuals)
    }


#Exponential Smoothing
def exponential_smoothing(series, alpha=0.3, forecast_steps=10):
    """
    Simple Exponential Smoothing for short-term forecasting.
    alpha: smoothing parameter (0 < alpha < 1)
    Higher alpha = more weight on recent observations
    """
    if hasattr(series, 'dropna'):
        series = series.dropna().values
    series = np.array(series)
    smoothed = np.zeros(len(series))
    smoothed[0] = series[0]

    for t in range(1, len(series)):
        smoothed[t] = alpha * series[t] + (1 - alpha) * smoothed[t-1]

    # Forecast
    last_smooth = smoothed[-1]
    forecasts = np.full(forecast_steps, last_smooth)

    return smoothed, forecasts


#EWMA Volatility
def ewma_volatility(returns, lambda_=0.94, window=252):
    """
    EWMA (Exponentially Weighted Moving Average) Volatility.
    RiskMetrics standard: lambda = 0.94 for daily data.
    More responsive to recent volatility than simple rolling std.
    """
    returns = returns.dropna().values
    n = len(returns)
    ewma_var = np.zeros(n)
    ewma_var[0] = returns[0]**2

    for t in range(1, n):
        ewma_var[t] = (lambda_ * ewma_var[t-1] +
                       (1 - lambda_) * returns[t]**2)

    ewma_vol = np.sqrt(ewma_var * window) * 100
    return ewma_vol


def run_time_series_analysis(price_df, returns_df, target_stocks=None):
    """Run full time series analysis pipeline."""
    if target_stocks is None:
        target_stocks = ['TCS', 'HDFCBANK', 'RELIANCE',
                         'HINDUNILVR', 'SUNPHARMA']

    results = {}

    print("\nRunning Time Series Analysis...")
    print("-" * 50)

    for stock in target_stocks:
        if stock not in price_df.columns:
            continue

        prices  = price_df[stock].dropna()
        returns = returns_df[stock].dropna()

        # 1. Stationarity test on prices
        adf_price = adf_test_simple(prices)

        # 2. Stationarity test on returns
        adf_returns = adf_test_simple(returns)

        # 3. ACF
        acf_vals, conf = compute_acf(returns, max_lags=10)

        # 4. ARIMA forecast
        model = SimpleARIMA(p=2, d=1, q=0)
        model.fit(prices.values)
        forecast_10 = model.forecast(prices.values, steps=10)

        # 5. Rolling forecast evaluation
        eval_metrics = rolling_forecast_evaluation(prices)

        # 6. EWMA Volatility
        ewma_vol = ewma_volatility(returns)

        # 7. Exponential Smoothing
        smoothed, es_forecast = exponential_smoothing(prices.values)

        results[stock] = {
            'adf_prices'    : adf_price,
            'adf_returns'   : adf_returns,
            'acf'           : acf_vals,
            'acf_conf'      : conf,
            'arima_forecast': forecast_10,
            'eval_metrics'  : eval_metrics,
            'ewma_vol'      : ewma_vol,
            'es_forecast'   : es_forecast,
            'last_price'    : prices.iloc[-1],
        }

        print(f"  {stock:12} | ADF Returns: {adf_returns['conclusion']:14} "
              f"| RMSE: {eval_metrics['RMSE']:.2f} "
              f"| Dir Acc: {eval_metrics['Dir_Accuracy']:.1f}%")

    print("-" * 50)
    print(f"Time series analysis complete for {len(results)} stocks")
    return results
