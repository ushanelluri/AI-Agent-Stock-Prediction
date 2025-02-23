#!/usr/bin/env python3
import os
import pandas as pd
import streamlit as st
from yahooquery import Ticker
import yfinance as yf
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables if needed (e.g., API keys)
load_dotenv()

def fetch_historical_data(ticker_symbol, period='1y', start_date=None, end_date=None):
    """
    Fetch historical stock data for a given ticker symbol.
    Uses a custom date range via yfinance if start_date and end_date are provided;
    otherwise, uses yahooquery with a preset period.
    """
    st.info(f"Fetching historical data for {ticker_symbol}...")
    if start_date and end_date:
        # Use yfinance for a custom date range
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
    else:
        # Use yahooquery for period-based historical data
        ticker = Ticker(ticker_symbol)
        data = ticker.history(period=period)
    
    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)
    else:
        st.error("Failed to fetch historical data as a DataFrame.")
        return None
    
    # Normalize column names to lowercase
    data.columns = data.columns.str.lower()
    
    # Ensure required columns exist
    required_columns = ['date', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns in historical data: {', '.join(missing_columns)}")
        return None
    return data

def fetch_realtime_data(ticker_symbol, period='1d', interval='1m'):
    """
    Fetch real-time stock data for a given ticker symbol using yfinance.
    Handles MultiIndex columns by flattening them.
    """
    st.info(f"Fetching real-time data for {ticker_symbol} (period={period}, interval={interval})...")
    data = yf.download(ticker_symbol, period=period, interval=interval)
    if data.empty:
        st.error("Failed to fetch real-time data.")
        return None
    data.reset_index(inplace=True)
    
    # If the columns are a MultiIndex, flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Convert column names to lowercase
    data.columns = data.columns.str.lower()
    
    # Rename datetime column to 'date' if present
    if 'datetime' in data.columns:
        data.rename(columns={'datetime': 'date'}, inplace=True)
    elif 'date' not in data.columns and data.index.name is not None:
        data.index.name = 'date'
        data.reset_index(inplace=True)
    
    # Ensure required columns exist
    required_columns = ['date', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns in real-time data: {', '.join(missing_columns)}")
        return None
    return data

def calculate_vpt(stock_data, calc_period=1, weighting_factor=1.0, apply_smoothing=False, smoothing_window=5):
    """
    Calculate the Volume Price Trend (VPT) indicator with customization options.
    - calc_period: period (in days) used for the percentage change calculation.
    - weighting_factor: factor to weight the volume * pct_change value.
    - apply_smoothing: if True, apply a rolling average to smooth the VPT.
    - smoothing_window: window size (in days) for the smoothing.
    """
    # Calculate percentage change based on the specified period
    stock_data['Price Change %'] = stock_data['close'].pct_change(periods=calc_period)
    # Compute the VPT with weighting factor and cumulative sum
    stock_data['VPT'] = (stock_data['volume'] * stock_data['Price Change %'] * weighting_factor).cumsum()
    
    # Optionally apply smoothing
    if apply_smoothing and smoothing_window > 1:
        stock_data['VPT'] = stock_data['VPT'].rolling(window=int(smoothing_window), min_periods=1).mean()
    
    return stock_data

def main():
    st.title("Volume Price Trend (VPT) Indicator")
    st.write("Customize and calculate the VPT indicator for your selected stock.")

    # Basic inputs for stock symbol and data source selection
    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    data_source = st.radio("Select Data Source:", options=["Historical Data", "Real-Time Data"])
    
    custom_date = False
    if data_source == "Historical Data":
        period_option = st.selectbox("Select Historical Data Period:", 
                                     options=["1y", "6mo", "3mo", "1mo", "Custom"], index=0)
        if period_option == "Custom":
            custom_date = True
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
    else:
        realtime_period = st.selectbox("Select Real-Time Data Period:", options=["1d", "5d"], index=0)
        realtime_interval = st.selectbox("Select Real-Time Data Interval:", options=["1m", "5m", "15m"], index=0)
    
    st.markdown("---")
    st.subheader("VPT Customization Options")
    calc_period = st.number_input("VPT Calculation Period (in days):", min_value=1, value=1, step=1)
    weighting_factor = st.number_input("Weighting Factor:", min_value=0.0, value=1.0, step=0.1)
    apply_smoothing = st.checkbox("Apply Smoothing to VPT")
    smoothing_window = st.number_input("Smoothing Window (in days):", min_value=1, value=5, step=1) if apply_smoothing else 5
    
    st.markdown("---")
    st.subheader("Chart Customization Options")
    display_chart = st.checkbox("Display VPT Chart", value=True)
    line_color = st.color_picker("Select Line Color", value="#0000FF")
    line_thickness = st.slider("Line Thickness", min_value=1, max_value=10, value=2)
    show_grid = st.checkbox("Show Grid", value=True)
    line_style = st.selectbox("Select Line Style:", options=["solid", "dashed", "dotted", "dashdot"], index=0)
    show_markers = st.checkbox("Show Markers", value=False)
    marker_size = st.slider("Marker Size", min_value=1, max_value=10, value=4) if show_markers else 0
    fig_width = st.slider("Chart Width (inches)", min_value=5, max_value=20, value=10)
    fig_height = st.slider("Chart Height (inches)", min_value=3, max_value=15, value=6)

    if st.button("Calculate VPT"):
        # Fetch data based on the selected data source and options
        if data_source == "Historical Data":
            if custom_date:
                data = fetch_historical_data(ticker_symbol, start_date=start_date, end_date=end_date)
            else:
                data = fetch_historical_data(ticker_symbol, period=period_option)
        else:
            data = fetch_realtime_data(ticker_symbol, period=realtime_period, interval=realtime_interval)
        
        if data is not None:
            st.subheader(f"Original Stock Data for {ticker_symbol}")
            st.dataframe(data.tail(10))
            
            # Calculate the VPT with customization options
            data_with_vpt = calculate_vpt(
                data,
                calc_period=calc_period,
                weighting_factor=weighting_factor,
                apply_smoothing=apply_smoothing,
                smoothing_window=smoothing_window
            )
            
            st.subheader("Stock Data with Calculated VPT")
            st.dataframe(data_with_vpt[['date', 'close', 'volume', 'VPT']].tail(20))
            
            # Plot the VPT chart if requested
            if display_chart:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                linestyle_map = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
                ls = linestyle_map.get(line_style, "-")
                
                if show_markers:
                    ax.plot(data_with_vpt['date'], data_with_vpt['VPT'], color=line_color,
                            linewidth=line_thickness, linestyle=ls, marker='o', markersize=marker_size)
                else:
                    ax.plot(data_with_vpt['date'], data_with_vpt['VPT'], color=line_color,
                            linewidth=line_thickness, linestyle=ls)
                
                ax.set_title(f"VPT Trend Over Time for {ticker_symbol}")
                ax.set_xlabel("Date")
                ax.set_ylabel("VPT")
                if show_grid:
                    ax.grid(True)
                st.pyplot(fig)

if __name__ == '__main__':
    main()
