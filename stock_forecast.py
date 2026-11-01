# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For stock data
import yfinance as yf

# For time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# For deep learning model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
# Download stock data (Example: Apple)
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")
print(data.head())

# Use only the 'Close' price
df = data[['Close']].reset_index()
df.columns = ['ds','y']  # Prophet needs columns as 'ds' (date) and 'y' (value)
plt.figure(figsize=(12,6))
plt.plot(df['ds'], df['y'], label="Stock Price")
plt.title("Apple Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Seasonal Decomposition
decomp = seasonal_decompose(df.set_index('ds')['y'], model='multiplicative', period=252) # ~1 year trading days
decomp.plot()
plt.show()
train = df.iloc[:-60]   # last 60 days for test
test = df.iloc[-60:]

model = ARIMA(train['y'], order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=60)

plt.figure(figsize=(12,6))
plt.plot(train['ds'], train['y'], label="Train")
plt.plot(test['ds'], test['y'], label="Test")
plt.plot(test['ds'], forecast, label="ARIMA Forecast")
plt.legend()
plt.show()
m = Prophet(daily_seasonality=True)
m.fit(train)

future = m.make_future_dataframe(periods=60)
forecast_prophet = m.predict(future)

fig1 = m.plot(forecast_prophet)
plt.show()

# Compare last 60 days
plt.figure(figsize=(12,6))
plt.plot(test['ds'], test['y'], label="Actual")
plt.plot(test['ds'], forecast_prophet['yhat'][-60:], label="Prophet Forecast")
plt.legend()
plt.show()
# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[['y']])

train_size = int(len(scaled_data)*0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Create sequences
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i+time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=32)

# Predict
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, label="Actual")
plt.plot(y_pred, label="LSTM Forecast")
plt.legend()
plt.show()
def evaluate(true, pred, name="Model"):
    rmse = sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# ARIMA
evaluate(test['y'].values, forecast.values, "ARIMA")

# Prophet
evaluate(test['y'].values, forecast_prophet['yhat'][-60:].values, "Prophet")

# LSTM
evaluate(y_test_rescaled, y_pred, "LSTM")
