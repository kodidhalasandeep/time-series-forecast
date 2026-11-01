#streamlit layout with tabs
import streamlit as st

st.set_page_config(page_title="TIME ETF Forecasting", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar ---
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

    # Define the chart here
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Actual",
        line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        mode="lines",
        name="Forecast",
        line=dict(color="orange", dash="dash")
    ))
    fig.update_layout(
        title=f"{model_choice} Forecast for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark" if dark_mode else "plotly_white"
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    # Plotly chart here (actual vs forecast)
    st.plotly_chart(fig, use_container_width=True)

    # Screenshot/export button
    st.download_button("📷 Export Chart as PNG", chart_image_bytes, file_name="forecast_chart.png")

    # CSV download
    st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False).encode("utf-8"), file_name="forecast.csv")
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
#Investor Simulation Logic
# Adjust forecast_days based on investor type
if investor_type == "Aggressive":
    forecast_days = 120
elif investor_type == "Conservative":
    forecast_days = 45