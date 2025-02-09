import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ==============================
# Gann Hi-Lo Activator Function
# ==============================
def calculate_gann_hi_lo_activator(df: pd.DataFrame, smoothing_period: int = 0) -> pd.DataFrame:
    """
    Calculates the Gann Hi-Lo Activator indicator.

    For each row:
      - If current close > previous activator:
            activator = min(current low, previous activator)
      - Otherwise:
            activator = max(current high, previous activator)

    Optionally applies exponential moving average (EMA) smoothing if smoothing_period > 1.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        smoothing_period (int): Period for EMA smoothing. If <= 1, no smoothing is applied.

    Returns:
        pd.DataFrame: Original DataFrame with two new columns:
                      - 'Gann Hi Lo': Raw indicator values.
                      - 'Gann Hi Lo Smoothed': Smoothed indicator values.
    """
    # Initialize the activator list with NaN values
    activator = [np.nan] * len(df)

    # Set the first value of the activator (commonly, the first low value)
    activator[0] = float(df['Low'].iloc[0])

    # Process each row sequentially
    for i in range(1, len(df)):
        current_close  = float(df['Close'].iloc[i])
        current_low    = float(df['Low'].iloc[i])
        current_high   = float(df['High'].iloc[i])
        prev_activator = float(activator[i - 1])

        # Determine the new activator based on directional movement
        if current_close > prev_activator:
            activator[i] = min(current_low, prev_activator)
        else:
            activator[i] = max(current_high, prev_activator)

    # Add calculated columns to the DataFrame
    df['Gann Hi Lo'] = activator

    if smoothing_period > 1:
        df['Gann Hi Lo Smoothed'] = pd.Series(activator, index=df.index)\
                                         .ewm(span=smoothing_period, adjust=False).mean()
    else:
        df['Gann Hi Lo Smoothed'] = df['Gann Hi Lo']

    return df

# ==============================
# Data Retrieval Functions
# ==============================
def fetch_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical market data for the given symbol between specified dates using yfinance.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def fetch_realtime_data(symbol: str) -> pd.DataFrame:
    """
    Fetch real-time intraday market data for the given symbol using yfinance.
    Real-time data is fetched as today's data with a 1-minute interval.
    """
    data = yf.download(symbol, period="1d", interval="1m")
    return data

def fetch_current_price(symbol: str):
    """
    Fetch the current stock price for the given symbol using yfinance ticker info.
    """
    ticker = yf.Ticker(symbol)
    current_price = ticker.info.get("regularMarketPrice")
    return current_price

# ==============================
# Streamlit UI
# ==============================
st.title("Gann Hi Lo Trading System")

# Initialize session state for storing fetched data
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()

# --- Stock Symbol Input ---
symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

# --- Data Mode Selection ---
data_mode = st.selectbox("Select Data Mode:", options=["Historical", "Real-Time"])

# ==============================
# Historical Data Section
# ==============================
if data_mode == "Historical":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"), key="start_date")
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("today"), key="end_date")
    
    if st.button("Fetch Historical Data"):
        st.session_state.data = fetch_historical_data(
            symbol, 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d")
        )
        if not st.session_state.data.empty:
            st.write(f"Historical Stock Data for {symbol}:")
            st.dataframe(st.session_state.data.tail())
        else:
            st.error("No data found for the selected date range.")

    # --- Gann Hi-Lo Calculation Section (Historical Data Only) ---
    if not st.session_state.data.empty:
        st.subheader("Gann Hi-Lo Customization Settings (Historical Data)")
        gann_smoothing = st.number_input(
            "Enter Gann Hi-Lo Smoothing Period:", 
            min_value=1, max_value=100, value=10, key="gann_smoothing_hist"
        )
        if st.button("Calculate Gann Hi-Lo Activator (Historical)"):
            data_with_gann = calculate_gann_hi_lo_activator(
                st.session_state.data.copy(), 
                smoothing_period=gann_smoothing
            )
            st.write(f"Historical Stock Data with Gann Hi-Lo Activator for {symbol}:")
            st.dataframe(data_with_gann.tail())

# ==============================
# Real-Time Data Section
# ==============================
else:
    # Toggle for auto-refreshing the current data every 5 seconds
    auto_refresh = st.checkbox("Auto Refresh Real-Time Data Every 5 Seconds", value=False)
    if auto_refresh:
        st_autorefresh(interval=5000, limit=0, key="real_time_autorefresh")

    # Button to fetch real-time data
    if st.button("Fetch Real-Time Data"):
        # Fetch the current stock price
        current_price = fetch_current_price(symbol)
        st.write(f"Current Stock Price for {symbol}: {current_price}")

        # Fetch real-time intraday data
        st.session_state.data = fetch_realtime_data(symbol)
        if not st.session_state.data.empty:
            st.write("Real-Time Intraday Data:")
            st.dataframe(st.session_state.data.tail())
        else:
            st.error("No real-time data found.")

    # --- Gann Hi-Lo Calculation Section for Real-Time Data ---
    if not st.session_state.data.empty:
        st.subheader("Gann Hi-Lo Customization Settings (Real-Time Data)")
        gann_smoothing_rt = st.number_input(
            "Enter Gann Hi-Lo Smoothing Period:", 
            min_value=1, max_value=100, value=10, key="gann_smoothing_rt"
        )
        if st.button("Calculate Gann Hi-Lo Activator (Real-Time)"):
            data_with_gann = calculate_gann_hi_lo_activator(
                st.session_state.data.copy(), 
                smoothing_period=gann_smoothing_rt
            )
            st.write(f"Real-Time Intraday Data with Gann Hi-Lo Activator for {symbol}:")
            st.dataframe(data_with_gann.tail())
