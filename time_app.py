#streamlit layout with tabs
import streamlit as st

st.set_page_config(page_title="TIME ETF Forecasting", layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("📊 TIME ETF Dashboard")
st.sidebar.markdown("Customize your view:")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
forecast_days = st.sidebar.slider("Forecast Horizon", 30, 180, 60)
investor_type = st.sidebar.selectbox("Investor Profile", ["Aggressive", "Conservative"])
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")
# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📐 Metrics", "🧠 Insights", "🤖 Ask AI"])
#Tab 1:Forecast(with screenshot and CSV Export)
import yfinance as yf
import pandas as pd

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df[["Date", "Close"]]
import datetime

ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

import plotly.graph_objects as go
with tab1:
    st.header("📈 Forecast Visualization")
    #Tab 3:Bussiness Impact Layer 
with tab3:
    st.header("🧠 Insights for Investors")

    show_insights = st.toggle("Show Business Takeaways", value=True)
    if show_insights:
        st.markdown("""
        ### 📌 Business Takeaways
        - Based on the current trend, **TIME ETF may enter a consolidation phase**.
        - **Volatility is increasing**, which may reduce **risk-adjusted returns**.
        - Conservative investors may prefer shorter horizons with Prophet forecasts.
        - Aggressive investors might explore LSTM for longer-term predictions.
        """)
    #Tab 4:Gemini AI Assistent
with tab4:
    st.header("🤖 Ask AI About TIME ETF")

    st.markdown("Try asking:")
    st.markdown("- *What sectors does TIME ETF invest in?*")
    st.markdown("- *How has TIME performed in the last 6 months?*")

    user_input = st.chat_input("Ask me anything about TIME ETF or forecasting...")
    if user_input:
        response = model.generate_content(user_input)
        st.markdown(response.text)
# ai_chat.py
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-pro")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []



# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
user_input = st.chat_input("Ask me anything about TIME ETF or stock forecasting...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.generate_content(user_input)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="INFY Stock Dashboard", layout="wide")
st.title("📈 INFY.NS Close Price Viewer")

# Download stock data
data = yf.download(
    "INFY.NS",
    start="2020-01-01",
    end="2025-06-30",
    progress=False
)[["Close"]]

# Display the data
st.subheader("Close Price Data (2020–2025)")
st.dataframe(data)

# Optional: Plot the closing prices
st.line_chart(data["Close"])
# Step 2: Convert index to datetime and fill missing business days
data.index = pd.to_datetime(data.index)
data = data.asfreq("B").ffill()

# Display the updated data
st.subheader("📅 Business Day-Aligned Close Prices")
st.write("Data has been reindexed to business days and forward-filled to handle gaps.")
st.dataframe(data)
st.line_chart(data["Close"])
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="INFY Stock Dashboard", layout="wide")
st.title("📊 INFY.NS Stock Data Sanity Check")

# Define ticker and time frame
ticker = "INFY.NS"
start_date = "2020-01-01"
end_date = "2025-06-30"

# Download full data
data = yf.download(ticker, start=start_date, end=end_date)

# Convert index to datetime
data.index = pd.to_datetime(data.index)

# Display basic info
st.write(f"✅ Rows: {data.shape[0]}")
st.write(f"✅ Frequency: {data.index.freq}")
st.subheader("📄 First 5 Rows")
st.dataframe(data.head())

st.subheader("📄 Last 5 Rows")
st.dataframe(data.tail())

# Missing values
st.subheader("🧹 Missing Values Per Column")
st.write(data.isnull().sum())

# Line plot of closing price
st.subheader("📈 Closing Price Over Time")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data["Close"], label="INFY.NS Close Price", color="blue")
ax.set_title("INFY.NS Close Price (2020-01-01 to 2025-06-30)")
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.legend()
st.pyplot(fig)
import streamlit as st
import matplotlib.pyplot as plt

st.subheader("📊 Rolling Statistics (30-Day Window)")

# Define rolling window
rolling_window = 30

# Compute rolling mean and standard deviation
rolling_mean = data["Close"].rolling(window=rolling_window).mean()
rolling_std = data["Close"].rolling(window=rolling_window).std()

# Plot using Streamlit-compatible rendering
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data["Close"], color="blue", label="Original Close Price")
ax.plot(rolling_mean, color="red", label=f"{rolling_window}-Day Rolling Mean")
ax.plot(rolling_std, color="black", label=f"{rolling_window}-Day Rolling Std")
ax.set_title("INFY.NS Close Price with Rolling Statistics")
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.legend()

# Display the plot in the dashboard
st.pyplot(fig)
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# 5. Seasonal Decomposition
st.subheader("🔍 Seasonal Decomposition of Close Price")

# 5.1 Decompose the series (additive model, annual seasonality ≈ 252 trading days)
decomposed = seasonal_decompose(data["Close"], model="additive", period=252)

# 5.2 Prepare a 4-panel plot
fig, (ax_orig, ax_trend, ax_seasonal, ax_resid) = plt.subplots(
    4, 1, sharex=True, figsize=(14, 10)
)

# 5.3 Plot each component
ax_orig.plot(data["Close"],        color="blue")
ax_orig.set_title("Original Series")

ax_trend.plot(decomposed.trend,    color="red")
ax_trend.set_title("Trend Component")

ax_seasonal.plot(decomposed.seasonal, color="green")
ax_seasonal.set_title("Seasonal Component")

ax_resid.plot(decomposed.resid,     color="black")
ax_resid.set_title("Residual Component")

plt.tight_layout()
ax_resid.set_xlabel("Date")

# 5.4 Render the figure in Streamlit
st.pyplot(fig)
import streamlit as st
from statsmodels.tsa.stattools import adfuller
import pandas as pd

st.subheader("🧪 Stationarity Test: Augmented Dickey–Fuller")

# Run the ADF test on the closing price series
adf_result = adfuller(data["Close"])

# Extract results
adf_stat  = adf_result[0]
p_value   = adf_result[1]
crit_vals = adf_result[4]

# Display the test statistic and p-value
st.write(f"**ADF Statistic:** {adf_stat:.6f}")
st.write(f"**p-value:** {p_value:.6f}")

# Display critical values in a table
cv_df = pd.DataFrame.from_dict(
    crit_vals, orient="index", columns=["Critical Value"]
).rename_axis("Confidence Level")
st.table(cv_df)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 6. First Differencing and ADF Re-test
st.subheader("🔄 First Differencing and ADF Re-test")

# 6.1 Create first-difference series
close_diff = data["Close"].diff().dropna()

# 6.2 Plot the differenced series
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(close_diff, color="purple")
ax.set_title("First Difference of INFY.NS Close Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price Change (INR)")
st.pyplot(fig)

# 6.3 Rerun ADF test on differenced data
adf_diff = adfuller(close_diff)
diff_stat  = adf_diff[0]
diff_pvalue = adf_diff[1]
diff_cv    = adf_diff[4]

# 6.4 Display ADF results
st.write(f"**Differenced ADF Statistic:** {diff_stat:.6f}")
st.write(f"**Differenced p-value:** {diff_pvalue:.6f}")

# 6.5 Show critical values in a table
diff_cv_df = pd.DataFrame.from_dict(
    diff_cv, orient="index", columns=["Critical Value"]
).rename_axis("Confidence Level")
st.table(diff_cv_df)
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.subheader("🔗 Autocorrelation (ACF) and Partial Autocorrelation (PACF)")

# Choose how many lags to display
lags = 30  # you can expose this as a slider via st.slider if you like

# Create a two‐panel plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ACF on the original closing price
plot_acf(data["Close"], lags=lags, ax=axes[0])
axes[0].set_title("ACF of Close Price")

# PACF on the first‐difference to identify AR terms
diff_series = data["Close"].diff().dropna()
plot_pacf(diff_series, lags=lags, ax=axes[1])
axes[1].set_title("PACF of First Difference")

# Render in Streamlit
st.pyplot(fig)
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.subheader("🔗 ACF and PACF of First-Difference")

# Optional: expose lags slider
lags = st.sidebar.slider("Number of lags", min_value=10, max_value=50, value=30, step=1)

# Create side-by-side ACF/PACF plots
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

plot_acf(close_diff, lags=lags, ax=axes[0])
axes[0].set_title("ACF of First-Difference")

plot_pacf(close_diff, lags=lags, ax=axes[1])
axes[1].set_title("PACF of First-Difference")

# Render the figure in Streamlit
st.pyplot(fig)
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.subheader("🔎 ACF and PACF for ARIMA Order Selection")

# Interactive slider to choose number of lags
lags = st.sidebar.slider("Number of lags", min_value=10, max_value=50, value=30)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ACF for MA(q) identification
plot_acf(close_diff, lags=lags, ax=axes[0])
axes[0].set_title("ACF of First-Difference")

# PACF for AR(p) identification
plot_pacf(close_diff, lags=lags, ax=axes[1], method="ywm")
axes[1].set_title("PACF of First-Difference")

plt.tight_layout()

# Render the figure in Streamlit
st.pyplot(fig)
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.subheader("🔧 Fit ARIMA Model and Residual Diagnostics")

# Allow user to select ARIMA orders in the sidebar
p = st.sidebar.number_input("AR order (p)", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("Differencing order (d)", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("MA order (q)", min_value=0, max_value=5, value=1, step=1)

# Fit the ARIMA(p,d,q) model on the Close series
model = ARIMA(data["Close"], order=(p, d, q))
arima_res = model.fit()

# Display model summary
st.text("📋 ARIMA Model Summary")
st.text(arima_res.summary().as_text())

# Plot diagnostics (residuals, QQ plot, etc.)
st.subheader("⚙️ Residual Diagnostics")
fig = arima_res.plot_diagnostics(figsize=(12, 8))
st.pyplot(fig)
import streamlit as st
import matplotlib.pyplot as plt
from arch import arch_model

st.subheader("⚡ Volatility Modeling with GARCH(1,1)")

# 1. Retrieve residuals from the fitted ARIMA model
resid = arima_res.resid.dropna()

# 2. Let user choose GARCH orders in the sidebar
p_garch = st.sidebar.number_input("GARCH lag order p", min_value=1, max_value=5, value=1, step=1)
q_garch = st.sidebar.number_input("GARCH lag order q", min_value=1, max_value=5, value=1, step=1)

# 3. Fit the GARCH(p,q) model on the residuals
garch = arch_model(resid, vol="Garch", p=p_garch, q=q_garch)
garch_res = garch.fit(disp="off")

# 4. Display the model summary
st.text("📋 GARCH Model Summary")
st.text(garch_res.summary().as_text())

# 5. Extract conditional volatility
cond_vol = garch_res.conditional_volatility

# 6. Plot estimated volatility
st.subheader("📈 Conditional Volatility Over Time")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(cond_vol, color="purple", label=f"GARCH({p_garch},{q_garch}) Volatility")
ax.set_title(f"GARCH({p_garch},{q_garch}) Conditional Volatility of ARIMA Residuals")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility (Std. Dev.)")
ax.legend()
st.pyplot(fig)
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

st.subheader("🔍 Standardized Residuals Diagnostics")

# 1. Extract standardized residuals from the GARCH model
std_resid = garch_res.std_resid

# 2. Plot the standardized residual series
st.markdown("**Standardized Residuals Over Time**")
fig1, ax1 = plt.subplots(figsize=(12, 3))
ax1.plot(std_resid, color="black")
ax1.axhline(0, color="red", linestyle="--", linewidth=1)
ax1.set_title("Standardized Residuals from GARCH Model")
ax1.set_ylabel("Std. Residual")
ax1.set_xlabel("Date")
st.pyplot(fig1)

# 3. Plot the ACF of the standardized residuals
st.markdown("**ACF of Standardized Residuals**")
fig2, ax2 = plt.subplots(figsize=(8, 4))
plot_acf(std_resid, lags=30, ax=ax2)
ax2.set_title("Autocorrelation of Standardized Residuals")
st.pyplot(fig2)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.subheader("🔮 Forecasting Future Volatility & Price Bands")
# 3️⃣ Build rolling windows (6-month ≈ 126 trading‐day slices)
#    Change test_size if you’d like a different hold-out length
test_size = 126
windows   = []

n = len(data)
for start in range(0, n - test_size, test_size):
    train_end = start + test_size - 1
    test_end  = min(train_end + test_size, n - 1)
    windows.append((train_end, test_end))

# Quick sanity
print(f"Built {len(windows)} windows, e.g. first = {windows[0]}")
# 4️⃣ Sidebar controls & single-window forecast UI

import streamlit as st

model_choice = st.sidebar.selectbox(
    "Choose forecasting model",
    ["ARIMA", "SARIMA", "Prophet", "LSTM"],
    key="model_choice"
)
window_idx = st.sidebar.slider(
    "Select window index",
    0, len(windows) - 1, 0,
    key="window_idx"
)
run = st.sidebar.button("Run Forecast", key="run_forecast_main")

if run:
    te, we       = windows[window_idx]
    train_series = data["Close"].iloc[: te+1]
    test_series  = data["Close"].iloc[te+1 : we+1]
    n_steps      = len(test_series)

    
    if model_choice == "ARIMA":
        preds = forecast_arima(train_series, n_steps)
        
    elif model_choice == "SARIMA":
        preds = forecast_sarima(train_series, n_steps)
    elif model_choice == "Prophet":
        preds = forecast_prophet(train_series, n_steps)
    else:
        preds = forecast_lstm(train_series, n_steps)

    # comparison DataFrame
    df_res = pd.DataFrame({
        "Actual":   test_series.values,
        "Forecast": preds
    }, index=test_series.index)

    # compute & display metrics
    rmse, mae, mape = compute_metrics(test_series.values, preds)
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "MAPE (%)"],
        "Value":  [rmse, mae, mape]
    }).set_index("Metric")

    st.subheader(f"{model_choice} | Window {window_idx}")
    st.line_chart(df_res)
    st.dataframe(df_res)
    st.subheader("Error Metrics")
    st.table(metrics_df)

# --- Sidebar controls ---
horizon = st.sidebar.number_input(
    "Forecast horizon (days)", min_value=1, max_value=90, value=30, step=1
)
z_mul = st.sidebar.slider(
    "Z-score multiplier for confidence bands", min_value=1.0, max_value=3.0, value=1.96
)

# 1. Generate variance forecasts from GARCH
vol_fc = garch_res.forecast(horizon=horizon)
var_fc = vol_fc.variance.iloc[-1]                # last row: next `horizon` variances
vol30 = np.sqrt(var_fc)                          # convert variance → std dev

# 2. Show forecasted volatility table
st.write(f"**Forecasted {horizon}-Day Volatility (Std. Dev.)**")
st.dataframe(vol30.rename("Volatility"))

# 3. ARIMA mean forecast for the same horizon
arima_fc = arima_res.get_forecast(steps=horizon)
mean_fc = arima_fc.predicted_mean

# 4. Align indices and build confidence bands
vol30.index = mean_fc.index
lower = mean_fc - z_mul * vol30
upper = mean_fc + z_mul * vol30

# 5. Plot historical, forecast, and volatility bands
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data["Close"], label="Historical Close", color="blue")
ax.plot(mean_fc, label="Mean Forecast", color="red")
ax.fill_between(mean_fc.index, lower, upper,
                color="pink", alpha=0.3, label=f"{int((1-z_mul/3)*100)}% Bands")
ax.set_title("ARIMA+GARCH: Price Forecast with Volatility Bands")
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.legend()
st.pyplot(fig)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Section title
st.subheader("🔮 Forecasting Future Volatility & Price Forecasts")

# Sidebar controls
horizon = st.sidebar.number_input(
    "Forecast horizon (days)", min_value=1, max_value=90, value=30, step=1,
    key="volatility_horizon"
)
z_multiplier = st.sidebar.slider(
    "Confidence interval multiplier", min_value=0.5, max_value=3.0, value=1.96, step=0.01,
    key="volatility_multiplier"
)

# 1. Generate variance forecasts from the fitted GARCH model
vol_fc = garch_res.forecast(horizon=horizon)
var_fc = vol_fc.variance.iloc[-1]              # last row: next `horizon` variances
std_fc = np.sqrt(var_fc)                      # convert variance → standard deviation

# 2. Display forecasted volatility
st.write(f"**Forecasted {horizon}-Day Volatility (Std. Dev.)**")
st.dataframe(std_fc.rename("Volatility"))

# 3. Generate ARIMA mean forecast for the same horizon
arima_fc = arima_res.get_forecast(steps=horizon)
mean_fc = arima_fc.predicted_mean

# 4. Align indices and build confidence bands
std_fc.index = mean_fc.index
lower_band = mean_fc - z_multiplier * std_fc
upper_band = mean_fc + z_multiplier * std_fc

# 5. Plot historical prices, mean forecast, and volatility bands
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data["Close"], label="Historical Close", color="blue")
ax.plot(mean_fc, label="Mean Forecast", color="red")
ax.fill_between(
    mean_fc.index,
    lower_band,
    upper_band,
    color="pink",
    alpha=0.3,
    label=f"{int((1 - (1 - 0.95)) * 100)}% Volatility Bands"
)
ax.set_title("ARIMA + GARCH: Price Forecast with Volatility Bands")
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.legend()

# 6. Render the figure in Streamlit
st.pyplot(fig)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model   # move this up with your other imports

st.subheader("🔎 ARIMA Residual Series")

# 1. Drop NaNs and extract residuals
residuals = arima_res.resid.dropna()

# 2. Show residuals in a table
st.markdown("**Residuals after dropping missing values**")
st.dataframe(residuals.to_frame(name="Residuals"))

# 3. Plot residuals over time
st.markdown("**Residuals Time Series**")
st.line_chart(residuals)

# 4. (Optional) Histogram of residuals
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(residuals, bins=30, color="skyblue", edgecolor="black")
ax.set_title("Histogram of ARIMA Residuals")
ax.set_xlabel("Residual")
ax.set_ylabel("Frequency")
st.pyplot(fig)
import streamlit as st
from arch import arch_model
import matplotlib.pyplot as plt

st.subheader("⚡ GARCH(1,1) with Student’s t Distribution")

# 1. Ensure residuals are defined
# residuals = arima_res.resid.dropna()

# 2. Define and fit the GARCH model with Student’s t errors
garch_t_model = arch_model(
    residuals,
    mean="Zero",
    vol="Garch",
    p=1,
    q=1,
    dist="t"       # Student’s t distribution for heavy tails
)
garch_t_res = garch_t_model.fit(disp="off")

# 3. Display the model summary
st.text("📋 GARCH(1,1) Model Summary with Student’s t")
st.text(garch_t_res.summary().as_text())

# 4. Plot conditional volatility series
st.markdown("**Conditional Volatility (Student’s t)**")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(garch_t_res.conditional_volatility, color="purple", label="Cond. Volatility")
ax.set_title("GARCH(1,1) Conditional Volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility")
ax.legend()
st.pyplot(fig)
import streamlit as st
import matplotlib.pyplot as plt

st.subheader("⚙️ Fit GARCH(1,1)-t Model")

# 1. Fit the model (already defined as garch_t_model)
garch_t_res = garch_t_model.fit(disp="off")

# 2. Display the model summary
st.text("📋 GARCH(1,1)-t Model Summary")
st.text(garch_t_res.summary().as_text())

# 3. Plot the conditional volatility
st.markdown("**Conditional Volatility Over Time**")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(garch_t_res.conditional_volatility, color="navy")
ax.set_title("GARCH(1,1)-t: Conditional Volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility (Std. Dev.)")
st.pyplot(fig)
import streamlit as st
import pandas as pd

st.subheader("🗓️ Prepare Data & Define Rolling Test Windows")

# 1. Ensure DateTimeIndex and business-day frequency
data.index = pd.to_datetime(data.index)
data = data.asfreq("B").ffill()

# 2. Sidebar control for test-window size (in trading days)
test_size = st.sidebar.number_input(
    "Test window size (days)", min_value=30, max_value=252, value=126, step=1
)

# 3. Compute total observations
n_obs = len(data)
st.write(f"Total observations: {n_obs}")

# 4. Generate (train_end, test_end) index pairs
windows = []
start = 0
while start + test_size < n_obs:
    train_end = start + test_size - 1
    test_end = min(train_end + test_size, n_obs - 1)
    windows.append((train_end, test_end))
    start += test_size

# 5. Convert index pairs to a DataFrame with dates for display
window_rows = []
for te, tse in windows:
    window_rows.append({
        "Train End #": te,
        "Test End #": tse,
        "Train End Date": data.index[te].date(),
        "Test End Date": data.index[tse].date()
    })
df_windows = pd.DataFrame(window_rows)

st.write(f"Number of rolling windows: {len(windows)}")
st.dataframe(df_windows)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your forecasting functions
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def forecast_arima(train, steps):
    model = ARIMA(train, order=(1,1,1)).fit()
    return model.forecast(steps=steps)

def forecast_sarima(train, steps):
    model = SARIMAX(
        train,
        order=(1,1,1),
        seasonal_order=(1,0,1,252),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    return model.forecast(steps=steps)

def forecast_prophet(train, steps):
    df = train.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    m = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05)
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq='B')
    fcst = m.predict(future)
    return fcst['yhat'].iloc[-steps:].values

def forecast_lstm(train, steps, scaler=None):
    # Scale
    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(train.values.reshape(-1,1))
    else:
        scaled = scaler.transform(train.values.reshape(-1,1))

    # Build sequences
    n_steps = 60
    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i-n_steps:i,0])
        y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Fit LSTM
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(n_steps,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Forecast iteratively
    input_seq = scaled[-n_steps:].reshape((1, n_steps, 1))
    preds = []
    for _ in range(steps):
        yhat = model.predict(input_seq, verbose=0)[0,0]
        preds.append(yhat)
        input_seq = np.append(input_seq[:,1:,:], [[yhat]], axis=1)

    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()


# --- Streamlit App Layout ---
st.set_page_config(page_title="📊 Rolling Window Forecasting", layout="wide")
st.title("Rolling Window Forecast Comparison")

# Assume `data` and `windows` are defined earlier in your app
# data: DataFrame with Date index and 'Close' column
# windows: list of (train_end_idx, test_end_idx) tuples

# Sidebar controls
model_name = st.sidebar.selectbox(
    "Choose forecasting model",
    ["ARIMA", "SARIMA", "Prophet", "LSTM"]
)

window_idx = st.sidebar.slider(
    "Select rolling window index",
    min_value=0,
    max_value=len(windows)-1,
    value=0
)

run = st.sidebar.button("Run Forecast", key="run_forecast_secondary")
if run:
    # Extract train/test indices and slices
    te, tse = windows[window_idx]
    train_series = data["Close"].iloc[: te+1]
    test_series  = data["Close"].iloc[te+1 : tse+1]
    n_steps      = len(test_series)

    # Dispatch to the chosen model
    if model_name == "ARIMA":
        preds = forecast_arima(train_series, n_steps)
    elif model_name == "SARIMA":
        preds = forecast_sarima(train_series, n_steps)
    elif model_name == "Prophet":
        preds = forecast_prophet(train_series, n_steps)
    else:
        preds = forecast_lstm(train_series, n_steps)

    # Build a comparison DataFrame
    df_res = pd.DataFrame({
        "Actual":   test_series.values,
        "Forecast": preds
    }, index=test_series.index)

    # Display results
    st.subheader(f"Model: {model_name} | Window #{window_idx}")
    st.line_chart(df_res)
    st.dataframe(df_res)
    import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1️⃣ Define and register in memory
def compute_metrics(actual, pred):
    """
    actual : array-like of true values
    pred   : array-like of forecasted values
    returns : tuple of (rmse, mae, mape)
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae  = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return rmse, mae, mape

# ... your rolling-window code above that produces:
#    test_series : pd.Series of true close prices
#    preds       : np.ndarray of forecasts
split_index = int(len(df) * 0.8)
train_series = df['Close'][:split_index]
test_series = df['Close'][split_index:]
preds = forecast_arima(train_series, len(test_series))

preds=forecast_arima(train_series,len(test_series))
test_series = data["Close"].iloc[train_end+1 : test_end+1]
rmse, mae, mape = compute_metrics(test_series.values, preds)

# 2️⃣ After forecasting, compute metrics
rmse, mae, mape = compute_metrics(test_series.values, preds)

# 3️⃣ Build a DataFrame for display
metrics_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "MAPE (%)"],
    "Value": [rmse, mae, mape]
})

# 4️⃣ Render in Streamlit
st.subheader("🏆 Forecast Performance Metrics")
st.table(metrics_df)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Page config
st.set_page_config(page_title="📊 Rolling Window Backtest", layout="wide")
st.title("📈 Rolling-Window Backtest: ARIMA, SARIMA, Prophet & LSTM")

# 2. Download & prepare data
data = yf.download("INFY.NS", start="2019-01-01", end="2025-08-01", progress=False)[["Close"]]
data.index = pd.to_datetime(data.index)
data = data.asfreq("B").ffill()

# 3. Forecast functions
def forecast_arima(train, steps):
    return ARIMA(train, order=(1,1,1)).fit().forecast(steps=steps)

def forecast_sarima(train, steps):
    model = SARIMAX(
        train,
        order=(1,1,1),
        seasonal_order=(1,0,1,252),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    return model.forecast(steps=steps)

def forecast_prophet(train, steps):
    df = train.reset_index().rename(columns={"Date":"ds","Close":"y"})
    m  = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05)
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq="B")
    fcst   = m.predict(future)
    return fcst["yhat"].iloc[-steps:].values

def forecast_lstm(train, steps):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.values.reshape(-1,1))
    n_steps = 60

    # build sequences
    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i-n_steps:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(-1, n_steps, 1)

    # fit LSTM
    model = Sequential([
        LSTM(50, activation="tanh", input_shape=(n_steps,1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # rolling forecast
    seq   = scaled[-n_steps:].reshape(1, n_steps, 1)
    preds = []
    for _ in range(steps):
        yhat = model.predict(seq, verbose=0)[0,0]
        preds.append(yhat)
        seq = np.concatenate([seq[:,1:,:], [[ [yhat] ]]], axis=1)

    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

# 4. Metrics function
def compute_metrics(actual, pred):
    y_true = np.array(actual).flatten()
    y_pred = np.array(pred).flatten()
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    mape   = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# 5. Sidebar inputs
test_size = st.sidebar.number_input(
    "Test window size (trading days)", min_value=30, max_value=252, value=126, step=1
)
run_backtest = st.sidebar.button("▶️ Run Backtest")

# 6. Build rolling windows
n_obs   = len(data)
windows = []
start   = 0
while start + test_size < n_obs:
    train_end = start + test_size - 1
    test_end  = min(train_end + test_size, n_obs - 1)
    windows.append((train_end, test_end))
    start += test_size

st.write(f"Total observations: {n_obs} | Rolling windows: {len(windows)}")

# 7. Execute backtest
if run_backtest:
    results = []
    for train_end, test_end in windows:
        train = data["Close"].iloc[: train_end+1]
        test  = data["Close"].iloc[train_end+1 : test_end+1]
        steps = len(test)

        fa = forecast_arima(train, steps)
        fs = forecast_sarima(train, steps)
        fp = forecast_prophet(train, steps)
        fl = forecast_lstm(train, steps)

        rmse_a, mae_a, mape_a = compute_metrics(test, fa)
        rmse_s, mae_s, mape_s = compute_metrics(test, fs)
        rmse_p, mae_p, mape_p = compute_metrics(test, fp)
        rmse_l, mae_l, mape_l = compute_metrics(test, fl)

        results.append({
            "window":        f"{train.index[0].date()}→{test.index[-1].date()}",
            "RMSE_ARIMA":    rmse_a, "MAE_ARIMA":  mae_a, "MAPE_ARIMA":  mape_a,
            "RMSE_SARIMA":   rmse_s, "MAE_SARIMA": mae_s, "MAPE_SARIMA": mape_s,
            "RMSE_PROP":     rmse_p, "MAE_PROP":   mae_p, "MAPE_PROP":   mape_p,
            "RMSE_LSTM":     rmse_l, "MAE_LSTM":   mae_l, "MAPE_LSTM":   mape_l
        })

    results_df = pd.DataFrame(results).set_index("window")

    st.subheader("🏆 Aggregated Performance Metrics")
    st.dataframe(results_df)

    st.subheader("📊 Average RMSE by Model")
    avg_rmse = results_df[["RMSE_ARIMA","RMSE_SARIMA","RMSE_PROP","RMSE_LSTM"]].mean()
    st.bar_chart(avg_rmse)
    import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 0. Page configuration
st.set_page_config(page_title="📊 Rolling-Window Backtest", layout="wide")
st.title("📈 Backtesting Forecast Models on INFY.NS")

# 1. Data download & preprocessing
@st.cache_data
def load_data():
    df = yf.download("INFY.NS",
                     start="2019-01-01",
                     end="2025-08-01",
                     progress=False)[["Close"]]
    df.index = pd.to_datetime(df.index)
    return df.asfreq("B").ffill()

data = load_data()
st.write("### Data Snapshot")
st.dataframe(data.head(), height=200)

# 2. Forecasting functions
def forecast_arima(train, steps):
    model = ARIMA(endog=train,
                  order=(1, 1, 1),
                  trend="n",
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit()
    return model.forecast(steps=steps)

def forecast_sarima(train, steps):
    model = SARIMAX(endog=train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, 252),
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)
    return model.forecast(steps=steps)

def forecast_prophet(train, steps):
    df = train.reset_index().rename(columns={train.index.name or "index": "ds", "Close": "y"})
    m  = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05)
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq="B")
    fcst   = m.predict(future)
    return fcst["yhat"].iloc[-steps:].values

def forecast_lstm(train, steps):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    n_steps = 60

    # prepare sequences
    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i - n_steps:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(-1, n_steps, 1)

    # fit LSTM
    model = Sequential([
        LSTM(50, activation="tanh", input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # rolling forecast
    seq, preds = scaled[-n_steps:].reshape(1, n_steps, 1), []
    for _ in range(steps):
        yhat = model.predict(seq, verbose=0)[0, 0]
        preds.append(yhat)
        seq = np.concatenate([seq[:, 1:, :], [[ [yhat] ]]], axis=1)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# 3. Performance metric
def compute_metrics(actual, pred):
    a, p = np.array(actual).flatten(), np.array(pred).flatten()
    rmse = np.sqrt(mean_squared_error(a, p))
    mae  = mean_absolute_error(a, p)
    mape = np.mean(np.abs((a - p) / a)) * 100
    return rmse, mae, mape

# 4. Build rolling-window indices
test_size = st.sidebar.number_input("Test window size (days)",  min_value=30, max_value=252, value=126)
n      = len(data)
windows = []
for start in range(0, n - test_size, test_size):
    te = start + test_size - 1
    se = min(te + test_size, n - 1)
    windows.append((te, se))

st.sidebar.write(f"Total obs: {n} → Rolling windows: {len(windows)}")
run_bt = st.sidebar.button("▶️ Run Backtest")

# 5. Backtest execution
if run_bt:
    results = []
    for te, se in windows:
        train = data["Close"].iloc[: te + 1]
        test  = data["Close"].iloc[te + 1 : se + 1]
        steps = len(test)

        fa = forecast_arima(train, steps)
        fs = forecast_sarima(train, steps)
        fp = forecast_prophet(train, steps)
        fl = forecast_lstm(train, steps)

        rmse_a, mae_a, mape_a = compute_metrics(test, fa)
        rmse_s, mae_s, mape_s = compute_metrics(test, fs)
        rmse_p, mae_p, mape_p = compute_metrics(test, fp)
        rmse_l, mae_l, mape_l = compute_metrics(test, fl)

        results.append({
            "window":        f"{train.index[0].date()}→{test.index[-1].date()}",
            "RMSE_ARIMA":    rmse_a, "MAE_ARIMA":  mae_a, "MAPE_ARIMA":  mape_a,
            "RMSE_SARIMA":   rmse_s, "MAE_SARIMA": mae_s, "MAPE_SARIMA": mape_s,
            "RMSE_PROPHET":  rmse_p, "MAE_PROPHET": mae_p, "MAPE_PROPHET": mape_p,
            "RMSE_LSTM":     rmse_l, "MAE_LSTM":   mae_l, "MAPE_LSTM":   mape_l
        })

    df_results = pd.DataFrame(results).set_index("window")

    # 6. Display aggregated metrics
    st.subheader("🏆 Backtest Results")
    st.dataframe(df_results, use_container_width=True)

    # 7. Bar chart of average RMSE
    avg_rmse = df_results[["RMSE_ARIMA","RMSE_SARIMA","RMSE_PROPHET","RMSE_LSTM"]].mean()
    st.subheader("📊 Average RMSE by Model")
    st.bar_chart(avg_rmse)

    # 8. Download button for raw results
    st.download_button(
        "⬇️ Download results as CSV",
        df_results.to_csv().encode("utf-8"),
        file_name="backtest_results.csv",
        mime="text/csv"
    )
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assume `results` list is already populated
# Build results_df
results_df = pd.DataFrame(results).set_index('window')

# Dashboard Title
st.subheader("🏁 Backtest Summary & Metrics")

# 1. Display full results table
st.header("🗂️ Full Backtest Results")
st.dataframe(results_df, use_container_width=True)

# 2. Compute average RMSE, MAE, MAPE
avg_rmse = results_df[['RMSE_ARIMA','RMSE_SARIMA','RMSE_PROP','RMSE_LSTM']].mean()
avg_mae  = results_df[['MAE_ARIMA','MAE_SARIMA','MAE_PROP','MAE_LSTM']].mean()
avg_mape = results_df[['MAPE_ARIMA','MAPE_SARIMA','MAPE_PROP','MAPE_LSTM']].mean()

avg_metrics = pd.DataFrame({
    'Metric': ['RMSE','MAE','MAPE (%)'],
    'ARIMA':  [avg_rmse.RMSE_ARIMA, avg_mae.MAE_ARIMA, avg_mape.MAPE_ARIMA],
    'SARIMA': [avg_rmse.RMSE_SARIMA, avg_mae.MAE_SARIMA, avg_mape.MAPE_SARIMA],
    'Prophet':[avg_rmse.RMSE_PROP,  avg_mae.MAE_PROP,  avg_mape.MAPE_PROP],
    'LSTM':   [avg_rmse.RMSE_LSTM,  avg_mae.MAE_LSTM,  avg_mape.MAPE_LSTM]
})

# 3. Display average metrics
st.header("📊 Average Metrics Across Windows")
st.table(avg_metrics.set_index('Metric'))

# 4. Plot bar chart of average RMSE
st.header("📈 Average RMSE Comparison")
fig, ax = plt.subplots(figsize=(8, 4))
models = ['ARIMA','SARIMA','Prophet','LSTM']
rmse_vals = [avg_rmse.RMSE_ARIMA, avg_rmse.RMSE_SARIMA, avg_rmse.RMSE_PROP, avg_rmse.RMSE_LSTM]
ax.bar(models, rmse_vals, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728'])
ax.set_title("Average RMSE by Model")
ax.set_ylabel("RMSE")
ax.set_xlabel("Model")
for i, v in enumerate(rmse_vals):
    ax.text(i, v + max(rmse_vals)*0.01, f"{v:.2f}", ha='center')
st.pyplot(fig)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.subheader("▶️ Single-Window Backtest Summary")

# 1. Select only the first rolling window
window = windows[0]
train_end, test_end = window
st.write(f"Testing on window: {data.index[0].date()} → {data.index[test_end].date()}")

# 2. Prepare train & test series
train = data["Close"].iloc[: train_end + 1]
test = data["Close"].iloc[train_end + 1 : test_end + 1]
steps = len(test)

# 3. Forecast for each model
fa = forecast_arima(train, steps)
fs = forecast_sarima(train, steps)
fp = forecast_prophet(train, steps)
fl = forecast_lstm(train, steps)

# 4. Compute metrics
def compute_metrics(actual, pred):
    a = np.array(actual).flatten()
    p = np.array(pred).flatten()
    rmse = np.sqrt(mean_squared_error(a, p))
    mae = mean_absolute_error(a, p)
    mape = np.mean(np.abs((a - p) / a)) * 100
    return rmse, mae, mape

metrics = {
    "ARIMA": compute_metrics(test, fa),
    "SARIMA": compute_metrics(test, fs),
    "Prophet": compute_metrics(test, fp),
    "LSTM": compute_metrics(test, fl)
}

# 5. Display results in a DataFrame
metrics_df = pd.DataFrame(metrics, index=["RMSE", "MAE", "MAPE"]).T

st.subheader("📊 Error Metrics")
st.dataframe(metrics_df.round(2))
import streamlit as st
import pandas as pd
import numpy as np

# Assume `data`, `windows`, forecasting functions, and `compute_metrics` are defined above

st.subheader("▶️ Single-Window Backtest")

# 1. Pick only the first rolling window for a quick smoke test
first_window = windows[:1]
te, se = first_window[0]
st.write(f"Testing on window: {data.index[0].date()} → {data.index[se].date()}")

# 2. Run forecasts and compute metrics
results = []
for train_end, test_end in first_window:
    train = data["Close"].iloc[: train_end + 1]
    test  = data["Close"].iloc[train_end + 1 : test_end + 1]
    steps = len(test)

    fa = forecast_arima(train, steps)
    fs = forecast_sarima(train, steps)
    fp = forecast_prophet(train, steps)
    fl = forecast_lstm(train, steps)

    rmse_a, mae_a, mape_a = compute_metrics(test, fa)
    rmse_s, mae_s, mape_s = compute_metrics(test, fs)
    rmse_p, mae_p, mape_p = compute_metrics(test, fp)
    rmse_l, mae_l, mape_l = compute_metrics(test, fl)

    results.append({
        "window":         f"{train.index[0].date()} → {test.index[-1].date()}",
        "RMSE_ARIMA":     rmse_a,
        "MAE_ARIMA":      mae_a,
        "MAPE_ARIMA (%)": mape_a,
        "RMSE_SARIMA":    rmse_s,
        "MAE_SARIMA":     mae_s,
        "MAPE_SARIMA (%)": mape_s,
        "RMSE_PROPHET":   rmse_p,
        "MAE_PROPHET":    mae_p,
        "MAPE_PROPHET (%)": mape_p,
        "RMSE_LSTM":      rmse_l,
        "MAE_LSTM":       mae_l,
        "MAPE_LSTM (%)":  mape_l,
    })

# 3. Build DataFrame and display
results_df = pd.DataFrame(results).set_index("window")

st.subheader("🏆 Backtest Metrics")
st.dataframe(results_df)
import streamlit as st
import pandas as pd
import numpy as np
import time

st.subheader("🔃 Single-Window Backtest with Progress")

# 1. Pick just the first window
window = windows[:1]  
te, we = window[0]
train_date = data.index[0].date()
test_date  = data.index[we].date()
st.write(f"Testing on window: {train_date} → {test_date}")

# 2. Prepare UI elements for progress
progress_bar = st.progress(0)
status_text  = st.empty()

# 3. Run backtest loop
results = []
start_all = time.perf_counter()

for i, (te, we) in enumerate(window, start=1):
    status_text.text(f"⏳ Window {i}/{len(window)}: running forecasts…")
    t0 = time.perf_counter()

    # slice train & test
    train = data["Close"].iloc[: te + 1]
    test  = data["Close"].iloc[te + 1 : we + 1]
    steps = len(test)

    # generate forecasts
    fa = forecast_arima(train, steps)
    fs = forecast_sarima(train, steps)
    fp = forecast_prophet(train, steps)
    fl = forecast_lstm(train, steps)

    # compute metrics
    rmse_a, mae_a, mape_a = compute_metrics(test, fa)
    rmse_s, mae_s, mape_s = compute_metrics(test, fs)
    rmse_p, mae_p, mape_p = compute_metrics(test, fp)
    rmse_l, mae_l, mape_l = compute_metrics(test, fl)

    # collect results
    results.append({
        "window":          f"{train_date} → {test_date}",
        "RMSE_ARIMA":      rmse_a,  "MAE_ARIMA":      mae_a,  "MAPE_ARIMA (%)": mape_a,
        "RMSE_SARIMA":     rmse_s,  "MAE_SARIMA":     mae_s,  "MAPE_SARIMA (%)": mape_s,
        "RMSE_PROPHET":    rmse_p,  "MAE_PROPHET":    mae_p,  "MAPE_PROPHET (%)": mape_p,
        "RMSE_LSTM":       rmse_l,  "MAE_LSTM":       mae_l,  "MAPE_LSTM (%)":   mape_l
    })

    # update progress
    elapsed = time.perf_counter() - t0
    status_text.text(f"✔ Window {i} done in {elapsed:.1f}s")
    progress_bar.progress(i / len(window))

# 4. Final timing
total_time = time.perf_counter() - start_all
status_text.text(f"✅ Completed in {total_time:.1f}s")

# 5. Build & display DataFrame
results_df = pd.DataFrame(results).set_index("window")
st.subheader("🏆 Backtest Results")
st.dataframe(results_df, use_container_width=True)
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Page configuration
st.set_page_config(page_title="📊 Rolling-Window Backtest", layout="wide")
st.title("🚀 Full Rolling-Window Backtest with Progress")

# 2. Load & prepare data (cached)
@st.cache_data
def load_data():
    df = (
        yf.download("INFY.NS",
                    start="2019-01-01",
                    end="2025-08-01",
                    progress=False)[["Close"]]
    )
    df.index = pd.to_datetime(df.index)
    return df.asfreq("B").ffill()

data = load_data()
st.subheader("🔍 Data Snapshot")
st.dataframe(data.head(), height=200)

# 3. Forecasting functions
def forecast_arima(train, steps):
    m = ARIMA(
        endog=train,
        order=(1, 1, 1),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit()
    return m.forecast(steps=steps)

def forecast_sarima(train, steps):
    m = SARIMAX(
        endog=train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 252),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    return m.forecast(steps=steps)

def forecast_prophet(train, steps):
    df = train.reset_index().rename(columns={train.index.name or "index": "ds", "Close": "y"})
    m = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05)
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq="B")
    fcst = m.predict(future)
    return fcst["yhat"].iloc[-steps:].values

def forecast_lstm(train, steps):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    n_steps = 60

    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i-n_steps:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, n_steps, 1)
    y = np.array(y)

    model = Sequential([
        LSTM(50, activation="tanh", input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    seq, preds = scaled[-n_steps:].reshape(1, n_steps, 1), []
    for _ in range(steps):
        yhat = model.predict(seq, verbose=0)[0, 0]
        preds.append(yhat)
        step = np.array(yhat).reshape(1, 1, 1)
        seq = np.concatenate([seq[:, 1:, :], step], axis=1)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# 4. Performance metric
def compute_metrics(actual, pred):
    a = np.array(actual).flatten()
    p = np.array(pred).flatten()
    rmse = np.sqrt(mean_squared_error(a, p))
    mae  = mean_absolute_error(a, p)
    mape = np.mean(np.abs((a - p) / a)) * 100
    return rmse, mae, mape

# 5. Define rolling windows
test_size = st.sidebar.number_input(
    "Test window size (trading days)", min_value=30, max_value=252, value=126, step=1
)
n_obs = len(data)
windows = []
for start in range(0, n_obs - test_size, test_size):
    te = start + test_size - 1
    we = min(te + test_size, n_obs - 1)
    windows.append((te, we))

st.sidebar.write(f"Total obs: {n_obs}  |  Windows: {len(windows)}")
run_bt = st.sidebar.button("▶️ Run Full Backtest")

# 6. Execute backtest
if run_bt:
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    overall_start = time.perf_counter()

    for i, (te, we) in enumerate(windows, start=1):
        status_text.text(f"⏳ Window {i}/{len(windows)} running…")
        t0 = time.perf_counter()

        train = data["Close"].iloc[: te + 1]
        test  = data["Close"].iloc[te + 1 : we + 1]
        steps = len(test)

        fa = forecast_arima(train, steps)
        fs = forecast_sarima(train, steps)
        fp = forecast_prophet(train, steps)
        fl = forecast_lstm(train, steps)

        r_a, m_a, mp_a = compute_metrics(test, fa)
        r_s, m_s, mp_s = compute_metrics(test, fs)
        r_p, m_p, mp_p = compute_metrics(test, fp)
        r_l, m_l, mp_l = compute_metrics(test, fl)

        results.append({
            "window":         f"{train.index[0].date()}→{test.index[-1].date()}",
            "RMSE_ARIMA":     r_a,   "MAE_ARIMA":     m_a,   "MAPE_ARIMA (%)": mp_a,
            "RMSE_SARIMA":    r_s,   "MAE_SARIMA":    m_s,   "MAPE_SARIMA (%)": mp_s,
            "RMSE_PROPHET":   r_p,   "MAE_PROPHET":   m_p,   "MAPE_PROPHET (%)": mp_p,
            "RMSE_LSTM":      r_l,   "MAE_LSTM":      m_l,   "MAPE_LSTM (%)":    mp_l
        })

        elapsed = time.perf_counter() - t0
        status_text.text(f"✔ Window {i} done in {elapsed:.1f}s")
        progress_bar.progress(i / len(windows))

    total_time = time.perf_counter() - overall_start
    status_text.text(f"✅ Completed full backtest in {total_time:.1f}s")

    # 7. Build & display DataFrame
    results_df = pd.DataFrame(results).set_index("window")
    st.subheader("🏆 Aggregated Backtest Results")
    st.dataframe(results_df, use_container_width=True)

    # 8. Bar chart of average RMSE by model
    avg_rmse = results_df[
        ["RMSE_ARIMA", "RMSE_SARIMA", "RMSE_PROPHET", "RMSE_LSTM"]
    ].mean()
    st.subheader("📈 Average RMSE by Model")
    st.bar_chart(avg_rmse)

    # 9. Download button for raw results
    st.download_button(
        "⬇️ Download results as CSV",
        results_df.to_csv().encode("utf-8"),
        file_name="backtest_results.csv",
        mime="text/csv"
    )
    import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- assume your backtest loop has populated `results` list ---

# 1. Build the DataFrame
results_df = pd.DataFrame(results).set_index("window")

# 2. Show completion header
st.subheader("🏁 Backtest Complete")

# 3. Display full results as a scrollable table
st.dataframe(results_df, use_container_width=True, height=400)

# 4. Compute average metrics across windows
avg_rmse = results_df[["RMSE_ARIMA", "RMSE_SARIMA", "RMSE_PROP", "RMSE_LSTM"]].mean()
avg_mae  = results_df[["MAE_ARIMA",  "MAE_SARIMA",  "MAE_PROP",  "MAE_LSTM"]].mean()
avg_mape = results_df[["MAPE_ARIMA", "MAPE_SARIMA", "MAPE_PROP", "MAPE_LSTM"]].mean()

# 5. Build a summary table
summary_df = pd.DataFrame({
    "Model":    ["ARIMA", "SARIMA", "Prophet", "LSTM"],
    "Avg RMSE": avg_rmse.values,
    "Avg MAE":  avg_mae.values,
    "Avg MAPE (%)": avg_mape.values
})

st.subheader("📊 Average Metrics Across Windows")
st.table(summary_df.set_index("Model"))

# 6. Plot average RMSE bar chart
st.subheader("📈 Average RMSE by Model")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(summary_df["Model"], summary_df["Avg RMSE"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
ax.set_ylabel("RMSE")
for i, v in enumerate(summary_df["Avg RMSE"]):
    ax.text(i, v + max(summary_df["Avg RMSE"]) * 0.01, f"{v:.2f}", ha="center")
st.pyplot(fig)
# 4B. Full backtest over every window
results = []
start_all = time.perf_counter()

for i, (te, we) in enumerate(windows, start=1):
    print(f" • Window {i}/{len(windows)}: train[0:{te+1}] → test[{te+1}:{we+1}]")
    t0 = time.perf_counter()

    train = data["Close"].iloc[: te+1]
    test  = data["Close"].iloc[te+1 : we+1]
    steps = len(test)

    # forecasts
    fa = forecast_arima(train, steps)
    fs = forecast_sarima(train, steps)
    fp = forecast_prophet(train, steps)
    fl = forecast_lstm(train, steps)

    # metrics
    rmse_a, mae_a, mape_a = compute_metrics(test, fa)
    rmse_s, mae_s, mape_s = compute_metrics(test, fs)
    rmse_p, mae_p, mape_p = compute_metrics(test, fp)
    rmse_l, mae_l, mape_l = compute_metrics(test, fl)

    results.append({
        "window":         f"{train.index[0].date()}→{test.index[-1].date()}",
        "RMSE_ARIMA":     rmse_a,  "MAE_ARIMA":     mae_a,  "MAPE_ARIMA (%)": mape_a,
        "RMSE_SARIMA":    rmse_s,  "MAE_SARIMA":    mae_s,  "MAPE_SARIMA (%)": mape_s,
        "RMSE_PROP":      rmse_p,  "MAE_PROP":      mae_p,  "MAPE_PROP (%)":   mape_p,
        "RMSE_LSTM":      rmse_l,  "MAE_LSTM":      mae_l,  "MAPE_LSTM (%)":   mape_l
    })

    print(f"   ✔ done in {time.perf_counter() - t0:.1f}s")

print(f"\n→ Completed full backtest in {time.perf_counter() - start_all:.1f}s\n")

# 5. Build & display DataFrame
results_df = pd.DataFrame(results).set_index("window")
print("→ Backtest complete:")
display(results_df)

# 6. Compute and print averages
avg_rmse = results_df[["RMSE_ARIMA","RMSE_SARIMA","RMSE_PROP","RMSE_LSTM"]].mean()
avg_mae  = results_df[["MAE_ARIMA","MAE_SARIMA","MAE_PROP","MAE_LSTM"]].mean()
avg_mape = results_df[["MAPE_ARIMA (%)","MAPE_SARIMA (%)","MAPE_PROP (%)","MAPE_LSTM (%)"]].mean()

print("Average RMSE:\n", avg_rmse, "\n")
print("Average MAE:\n",  avg_mae,  "\n")
print("Average MAPE:\n", avg_mape, "\n")

# 7. Quick bar chart of average RMSE
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,4))
avg_rmse.plot.bar(ax=ax, rot=0, title="Average RMSE by Model")
ax.set_ylabel("RMSE")
plt.show()
import streamlit as st
import pandas as pd
import numpy as np

# assume `data`, `forecast_arima`, `forecast_sarima`, 
# `forecast_prophet`, `forecast_lstm`, and `compute_metrics` are already defined

st.subheader("🔍 Smoke Test vs. Full Backtest")

# Sidebar controls
run_smoke = st.sidebar.checkbox("Run smoke test (1 window)", value=True)
test_size = st.sidebar.number_input(
    "Window size (trading days)", min_value=30, max_value=252, value=126, step=1
)
run_bt = st.sidebar.button("▶️ Run Backtest")

# Build rolling windows
n = len(data)
windows = []
for start in range(0, n - test_size, test_size):
    te = start + test_size - 1
    we = min(te + test_size, n - 1)
    windows.append((te, we))

st.sidebar.write(f"Total windows: {len(windows)}")

if run_bt:
    # Select windows
    if run_smoke:
        st.write("### 🧪 Smoke Test (first window only)")
        selected = windows[:1]
    else:
        st.write("### 🚀 Full Backtest (all windows)")
        selected = windows

    # Prepare results container
    results = []
    progress = st.progress(0)
    status = st.empty()

    # Loop through windows
    for i, (te, we) in enumerate(selected, start=1):
        # Update status
        status.text(f"⏳ Window {i}/{len(selected)}")
        
        # Slice train/test
        train = data["Close"].iloc[: te + 1]
        test  = data["Close"].iloc[te + 1 : we + 1]
        steps = len(test)

        # Forecast
        fa = forecast_arima(train, steps)
        fs = forecast_sarima(train, steps)
        fp = forecast_prophet(train, steps)
        fl = forecast_lstm(train, steps)

        # Compute metrics
        rmse_a, mae_a, mape_a = compute_metrics(test, fa)
        rmse_s, mae_s, mape_s = compute_metrics(test, fs)
        rmse_p, mae_p, mape_p = compute_metrics(test, fp)
        rmse_l, mae_l, mape_l = compute_metrics(test, fl)

        # Collect
        results.append({
            "window":           f"{train.index[0].date()}→{test.index[-1].date()}",
            "RMSE_ARIMA":       rmse_a,  "MAE_ARIMA":     mae_a,  "MAPE_ARIMA (%)": mape_a,
            "RMSE_SARIMA":      rmse_s,  "MAE_SARIMA":    mae_s,  "MAPE_SARIMA (%)": mape_s,
            "RMSE_PROPHET":     rmse_p,  "MAE_PROPHET":   mae_p,  "MAPE_PROPHET (%)": mape_p,
            "RMSE_LSTM":        rmse_l,  "MAE_LSTM":      mae_l,  "MAPE_LSTM (%)":    mape_l
        })
        progress.progress(i / len(selected))

    status.text("✅ Backtest complete")
    
    # Build results DataFrame
    results_df = pd.DataFrame(results).set_index("window")

    # Display full table
    st.subheader("📊 Backtest Results")
    st.dataframe(results_df, use_container_width=True)

    # Compute and display summary averages
    avg_rmse = results_df[["RMSE_ARIMA","RMSE_SARIMA","RMSE_PROPHET","RMSE_LSTM"]].mean()
    avg_mae  = results_df[["MAE_ARIMA","MAE_SARIMA","MAE_PROPHET","MAE_LSTM"]].mean()
    avg_mape = results_df[["MAPE_ARIMA (%)","MAPE_SARIMA (%)","MAPE_PROPHET (%)","MAPE_LSTM (%)"]].mean()

    summary = pd.DataFrame({
        "Model":      ["ARIMA","SARIMA","Prophet","LSTM"],
        "Avg RMSE":   avg_rmse.values,
        "Avg MAE":    avg_mae.values,
        "Avg MAPE (%)": avg_mape.values
    }).set_index("Model")

    st.subheader("📈 Average Metrics Across Windows")
    st.table(summary)

    # Bar chart of Avg RMSE
    st.subheader("📈 Average RMSE Comparison")
    st.bar_chart(avg_rmse)
    import streamlit as st
import pandas as pd
import numpy as np

# Assume data, forecast_*, compute_metrics are already defined

st.title("Rolling-Window Backtest: Smoke Test vs Full Run")

# Sidebar options
mode = st.sidebar.radio("Mode", ("Smoke Test (1 window)", "Full Backtest"))
test_size = st.sidebar.number_input("Window size (days)", 30, 252, 126)
run = st.sidebar.button("▶️ Run")

# Build windows
n = len(data)
windows = [(start + test_size - 1, min(start + 2*test_size - 1, n-1))
           for start in range(0, n - test_size, test_size)]

if run:
    selected = windows[:1] if mode.startswith("Smoke") else windows

    st.write(f"Running {'smoke test' if len(selected)==1 else 'full backtest'} on {len(selected)} window(s)")
    progress = st.progress(0)
    
    all_results = []
    for i, (te, we) in enumerate(selected, start=1):
        train = data["Close"].iloc[:te+1]
        test = data["Close"].iloc[te+1:we+1]
        steps = len(test)

        fa = forecast_arima(train, steps)
        fs = forecast_sarima(train, steps)
        fp = forecast_prophet(train, steps)
        fl = forecast_lstm(train, steps)

        # compute metrics
        metrics = {
            "window": f"{train.index[0].date()}→{test.index[-1].date()}"
        }
        for name, pred in [("ARIMA", fa), ("SARIMA", fs), ("Prophet", fp), ("LSTM", fl)]:
            rmse, mae, mape = compute_metrics(test, pred)
            metrics.update({
                f"RMSE_{name}": rmse,
                f"MAE_{name}": mae,
                f"MAPE_{name}": mape
            })
        all_results.append(metrics)
        progress.progress(i / len(selected))

    df_res = pd.DataFrame(all_results).set_index("window")
    st.subheader("Backtest Results")
    st.dataframe(df_res)

    avg = pd.DataFrame({
        "Avg RMSE": df_res.filter(like="RMSE_").mean(),
        "Avg MAE": df_res.filter(like="MAE_").mean(),
        "Avg MAPE (%)": df_res.filter(like="MAPE_").mean()
    })
    st.subheader("Average Metrics")
    st.table(avg)
    import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Assume you’ve already run your data download/prep and fitted your forecasts
# `df`          : pd.DataFrame with your INFY.NS data and a business-day index
# `forecast_df` : pd.DataFrame of your multi-model forecasts (columns per model)
# `model`       : the fitted statsmodels/Prophet/keras model you’d like to inspect
# compute_metrics, forecast_arima, forecast_sarima, forecast_prophet,
# forecast_lstm are all available in memory
# ─────────────────────────────────────────────────────────────────────────────

def show_data_section(title: str, data: pd.DataFrame):
    """Render top/bottom rows and missing‐value counts for a DataFrame."""
    st.subheader(title)
    col1, col2 = st.columns(2)
    with col1:
        st.write("First 5 rows")
        st.dataframe(data.head())
    with col2:
        st.write("Last  5 rows")
        st.dataframe(data.tail())
    st.write("Missing values per column:")
    st.write(data.isnull().sum())
    st.markdown("---")


# 1. STOCK DATA OVERVIEW
with st.container():
    show_data_section("📊 INFY.NS Close Price Overview", df)

# 2. FORECAST SUMMARY
with st.container():
    st.subheader("📈 Forecast Summary")
    st.write(
        "Below is the forecasted Close price for each model over your test window."
    )
    st.dataframe(forecast_df)  # forecast_df indexed by date, columns = model names
    st.markdown("---")

# 3. MODEL DIAGNOSTICS
with st.container():
    st.subheader("🛠️ Model Diagnostics")
    st.write("Examine the fitted parameters, residuals, and goodness-of-fit below.")
    st.text(model.summary().as_text())
    st.markdown("---")

# 4. OPTIONAL: PLOT ACTUAL vs FORECAST
with st.container():
    st.subheader("🔍 Actual vs Forecast Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Close"], label="Actual", color="black", linewidth=2)
    for col in forecast_df.columns:
        ax.plot(forecast_df.index, forecast_df[col], label=col, alpha=0.7)
    ax.set_title("Actual Close Price vs Model Forecasts")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    st.pyplot(fig)
#Footer Signature 
st.markdown("---")
st.markdown("🖋️ *Built by Sandeep, aspiring data scientist and product thinker. Let’s forecast the future together.*")