import yfinance as yf
import pandas as pd
import streamlit as st
import datetime
import os
import sys

# Adjust the system path so that our modules can be imported (if needed).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    """
    Fetch historical stock data for the specified date range and interval.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data

def calculate_atr(stock_data, period=14):
    """
    Calculate the Average True Range (ATR) for the given stock data and period.
    """
    # Calculate the True Range components
    stock_data['High-Low'] = stock_data['High'] - stock_data['Low']
    stock_data['High-Close'] = abs(stock_data['High'] - stock_data['Close'].shift(1))
    stock_data['Low-Close']  = abs(stock_data['Low']  - stock_data['Close'].shift(1))

    # True Range is the max of the three
    stock_data['True Range'] = stock_data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)

    # ATR is the rolling average of the True Range
    stock_data['ATR'] = stock_data['True Range'].rolling(window=period).mean()

    return stock_data

# Streamlit UI
st.title("Stock Data and ATR Calculator with Customization")

# 1. User input for Stock Ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "AAPL")

# 2. User-selectable date range
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.date(2022, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime.date.today())

# 3. Interval selection
interval_options = ["1d", "1wk", "1mo"]  # You can add more if desired (e.g., '1h', '5m', etc.)
selected_interval = st.selectbox("Select Data Interval", options=interval_options, index=0)

# 4. ATR Period slider
atr_period = st.slider("ATR Period", min_value=1, max_value=60, value=14, step=1)

# 5. ATR Alert Threshold
atr_threshold = st.number_input(
    "ATR Alert Threshold (optional)",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="If the ATR exceeds this value, a warning will be displayed."
)

# 6. Columns to Display
available_columns = ["Open", "High", "Low", "Close", "Volume", "ATR"]
selected_columns = st.multiselect(
    "Select Columns to Display",
    options=available_columns,
    default=["High", "Low", "Close", "ATR"]
)

# Fetch Stock Data button
if st.button('Fetch Stock Data'):
    if ticker:
        stock_data = fetch_stock_data(ticker, start_date, end_date, selected_interval)
        if not stock_data.empty:
            # Save to session state
            st.session_state.stock_data = stock_data
            st.write(f"Data successfully fetched for {ticker}. Here are the first 5 rows:")
            st.write(stock_data.head())
        else:
            st.warning("No data was returned. Please check your date range or ticker symbol.")
    else:
        st.error("Please enter a valid stock ticker.")

# Calculate ATR button
if st.button('Calculate ATR'):
    if 'stock_data' in st.session_state and not st.session_state.stock_data.empty:
        # Calculate ATR with user-defined period
        stock_data_with_atr = calculate_atr(st.session_state.stock_data.copy(), period=atr_period)

        # Check if ATR has exceeded the threshold
        latest_atr = stock_data_with_atr['ATR'].dropna().iloc[-1] if not stock_data_with_atr['ATR'].dropna().empty else 0.0
        if atr_threshold > 0 and latest_atr > atr_threshold:
            st.warning(f"Alert: The current ATR ({latest_atr:.2f}) exceeds the threshold of {atr_threshold}")

        # Display the last 5 rows of selected columns
        st.write(f"Displaying Stock Data and ATR for {ticker} with a period of {atr_period}:")
        columns_to_show = [col for col in selected_columns if col in stock_data_with_atr.columns]
        if columns_to_show:
            st.write(stock_data_with_atr[columns_to_show].tail())
        else:
            st.info("No columns selected to display.")

        # Plot ATR if requested
        if "ATR" in columns_to_show:
            st.line_chart(stock_data_with_atr["ATR"].dropna())
    else:
        st.error("Please fetch stock data first.")
