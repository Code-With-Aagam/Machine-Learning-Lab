```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```


```python
df = pd.read_csv('BAJAJFINSV.csv')

df.head()
```


```python
df.info()
```


```python
df.shape
```


```python
df.describe()
```


```python
df.isnull().sum()
```


```python
df.duplicated().sum()
```


```python
print("\nDate range:", df['Date'].min(), "to", df['Date'].max())
```


```python
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Close Price Over Time')
plt.legend()
plt.grid(True)
plt.show()
```


```python
from statsmodels.tsa.stattools import adfuller

df.set_index('Date',inplace=True)
ts = df['Close']
```


```python
rolling_mean = ts.rolling(window = 30).mean()
rolling_std = ts.rolling(window=30).std()
```


```python
plt.figure(figsize=(12,6))
plt.plot(ts,label = 'Original Series',color = 'blue')
plt.plot(rolling_mean,label = 'Rolling Mean(30 days)',color ='red')
plt.plot(rolling_std, label='Rolling Std (30 days)', color='green')
plt.title('Rolling Mean & Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


```python
adf_result = adfuller(ts.dropna())
print("ADF Test Results:")
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value}")
```


```python
ts_diff = ts.diff().dropna()
plt.figure(figsize=(12,6))
plt.plot(ts_diff, color='purple')
plt.title('First-order Differenced Close Price Series')
plt.xlabel('Date')
plt.ylabel('Differenced Price')
plt.grid(True)
plt.tight_layout()
plt.show()
```


```python
adf_diff_result = adfuller(ts_diff)
print("ADF Test Results (After First Differencing):")
print(f"ADF Statistic: {adf_diff_result[0]}")
print(f"p-value: {adf_diff_result[1]}")
print("Critical Values:")
for key, value in adf_diff_result[4].items():
    print(f"   {key}: {value}")
```


```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
```


```python
# --- Trend and Seasonality Decomposition ---
decomposition = seasonal_decompose(ts, model='additive', period=30)

plt.figure(figsize=(14,10))
plt.subplot(411)
plt.plot(decomposition.observed, label='Observed')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals', color='red')
plt.legend(loc='upper left')
plt.suptitle('Trend, Seasonality, and Residuals - Additive Decomposition', fontsize=16)
plt.tight_layout()
plt.show()
```

# Time Series Decomposition Analysis

The charts display an additive decomposition of a time series from 2008 to 2020, breaking it down into its fundamental components:

1. **Observed Data (Blue, Top Chart)**
   This is the original time series showing the raw data values over time. The values range from near 0 to about 10,000. There's a clear upward trend over the years, with a significant increase starting around 2016. There appears to be a notable drop around early 2020 (likely related to the COVID-19 pandemic) followed by a recovery.

2. **Trend Component (Yellow, Second Chart)**
   This isolates the long-term progression of the series after removing seasonality and noise. The trend shows a steady, gradual increase from 2008 to 2014, followed by a more pronounced upward trajectory from 2015 to 2018. Around 2020, there's a significant dip (matching the observed data) followed by a recovery to previous levels.

3. **Seasonality Component (Green, Third Chart)**
   This shows the regular, periodic fluctuations in the data. The consistent pattern of peaks and valleys indicates a regular seasonal cycle that repeats throughout the entire time period. The amplitude of these seasonal variations appears relatively stable, generally ranging between -20 and +10 units.

4. **Residuals (Red, Bottom Chart)**
   These represent the irregular or random variations in the data after removing trend and seasonality. Residuals should ideally look like random noise without patterns. In this case, the residuals appear relatively small and random until around 2016-2017, when they become larger and more volatile. This suggests that the variability in the data increased in later years, with a particularly large negative spike visible around 2020.


```python
# --- Autocorrelation and Partial Autocorrelation ---
plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
plot_acf(ts.dropna(), lags=50, ax=plt.gca())
plt.title('Autocorrelation (ACF)')

plt.subplot(1, 2, 2)
plot_pacf(ts.dropna(), lags=50, ax=plt.gca(), method='ywm')
plt.title('Partial Autocorrelation (PACF)')

plt.tight_layout()
plt.show()
```


```python
# Only convert and set index if 'Date' is still a column
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

# Extract Close price and handle missing values
ts = df['Close'].asfreq('B').fillna(method='ffill')

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(ts.diff().dropna(), ax=axes[0], lags=40)
plot_pacf(ts.diff().dropna(), ax=axes[1], lags=40)
axes[0].set_title('Autocorrelation Function (ACF)')
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()
```


```python
from statsmodels.tsa.arima.model import ARIMA
```


```python
model = ARIMA(ts, order = (2,1,2))
model_fit = model.fit()
```


```python
print(model_fit.summary())
```

## ARIMA Model Results

The fitted ARIMA model shows:

* **AR(1) & AR(2)** terms (`ar.L1`, `ar.L2`) and
* **MA(1) & MA(2)** terms (`ma.L1`, `ma.L2`) are all statistically significant (*p-values < 0.05*).
* **AIC = 39902.161** and **BIC = 39932.781**: These metrics are used for model comparison (lower is better).
* `sigma2` is the estimated variance of the residuals.
* **Warning:** *ConvergenceWarning* suggests the model took longer to fit or had difficulty converging — not unusual for complex ARIMA models, but something to watch if model diagnostics perform poorly.


```python
forecast_step = 30
forecast = model_fit.get_forecast(steps = forecast_step)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()
```


```python
plt.figure(figsize=(14, 6))
plt.plot(ts, label='Actual', color='blue')
plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_mean.index, 
                 conf_int.iloc[:, 0], 
                 conf_int.iloc[:, 1], 
                 color='pink', alpha=0.3, label='Confidence Interval')

plt.title('ARIMA Forecast vs Actual (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()
```

## Stock Price Forecast Analysis

* **Upward Trend (2008–2021):**
  * The stock price shows long-term **growth**, with some volatility — especially around 2020 (likely due to COVID-19 crash and rebound).
* **Forecast Segment:**
  * The forecast (red) starts where the actual data ends (~late 2021).
  * It **continues the upward trend**, reflecting what the ARIMA model "learned" from past data.
* **Confidence Interval:**
  * The pink shaded area grows wider — this is common in time series forecasting.
  * It shows that the model becomes **less certain** the further it predicts into the future.


```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
```


```python
pred = model_fit.predict(start =1,end = len(ts)-1,typ = 'levels')
actual = ts[1:]
rmse = np.sqrt(mean_squared_error(actual, pred))
mae = mean_absolute_error(actual, pred)
mape = np.mean(np.abs((actual - pred) / actual)) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
```

## Forecast Accuracy Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **RMSE** | `89.36` | On average, the prediction deviates from the actual value by ~89 units. Lower is better. |
| **MAE** | `41.72` | The average **absolute** difference between predicted and actual prices. Very useful and interpretable. |
| **MAPE** | `1.66%` | The average prediction error is only about **1.66%** of the actual value — this is **very accurate** for a stock model. |


```python
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.optimizers import Adam
```


```python
close_data = df[['Close']].values
```


```python
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)
```


```python
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)
```


```python
time_steps = 60
total_points = len(scaled_data)
```


```python
if total_points <= time_steps + 10:
    raise ValueError("Dataset too small to create sequences for LSTM. Add more data.")
```


```python
train_size = total_points - time_steps
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - time_steps:]
```


```python

```


```python
if X_test.size > 0:
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print("Shapes after reshape:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
else:
    raise ValueError("X_test is empty after sequence creation. Cannot proceed to training.")
```


```python
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
```


```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
```


```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)
```


```python
model.summary()
```


```python
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'],label ='Training Loss',color = 'blue')
plt.plot(history.history['val_loss'],label ='Validation Loss',color = 'red')
plt.title("Training Vs Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


```python
y_pred_scaled = model.predict(X_test)
```


```python
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
```


```python
rmse = np.sqrt(mean_squared_error(y_actual,y_pred))
mae = mean_absolute_error(y_actual,y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
```


```python
print(f"Evaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
```

## Stock Price Model Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **RMSE** | 418.81 | On average, predictions deviate by ~₹418 from actual values. RMSE penalizes large errors. |
| **MAE** | 337.63 | Average absolute error is ~₹337.63 — this is a direct indication of deviation. |
| **MAPE** | 3.38% | Your model is, on average, **96.6% accurate** in predicting stock price! This is very good. |


```python
plt.figure(figsize=(12, 6))
plt.plot(y_actual, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted (LSTM)', color='red')
plt.title('LSTM: Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()
```


```python

```
