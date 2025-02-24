import yfinance as yf
import pandas as pd
import streamlit as st

# Function to fetch stock data from Yahoo Finance based on data mode
def fetch_stock_data(ticker, data_mode, start_date=None, end_date=None, period=None, interval=None):
    """
    Fetch market data from Yahoo Finance.

    For historical data mode:
      - Uses start_date and end_date.

    For real-time data mode:
      - Uses period and interval (e.g., period="1d", interval="1m").

    :param ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
    :param data_mode: "Historical" or "Real-Time"
    :param start_date: Start date for historical data (datetime)
    :param end_date: End date for historical data (datetime)
    :param period: Period for real-time data (e.g., "1d", "5d")
    :param interval: Data interval for real-time data (e.g., "1m", "5m", "15m")
    :return: DataFrame containing stock data
    """
    if data_mode == "Real-Time":
        stock_data = yf.download(ticker, period=period, interval=interval)
    else:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
    st.write(f"Fetched {len(stock_data)} rows of data.")  # Debug: display row count
    return stock_data

# Helper function to calculate the selected moving average
def calculate_moving_average(series, period, ma_type):
    """
    Calculate a moving average on a given series.

    :param series: Pandas Series (e.g., Close prices)
    :param period: Number of periods
    :param ma_type: Type of moving average ("EMA" or "SMA")
    :return: Series with the calculated moving average
    """
    if ma_type == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    elif ma_type == "SMA":
        return series.rolling(window=period).mean()
    else:
        return series.ewm(span=period, adjust=False).mean()  # Default to EMA

# Function to calculate Elder-Ray Index values (Bull Power and Bear Power)
def calculate_elder_ray_index(stock_data, ma_period=14, ma_type="EMA", price_column="Close"):
    """
    Calculate the Elder-Ray Index values using the selected moving average type and period.

    :param stock_data: DataFrame containing market data (must include 'High', 'Low', and selected price_column)
    :param ma_period: Period for calculating the moving average
    :param ma_type: Type of moving average ("EMA" or "SMA")
    :param price_column: Price column to base the moving average on (default "Close")
    :return: DataFrame with additional columns for moving average, Bull Power, and Bear Power
    """
    # Calculate the moving average (MA) using the selected price column
    stock_data['MA'] = calculate_moving_average(stock_data[price_column], ma_period, ma_type)
    
    # Ensure that the columns 'High', 'MA', and 'Low' are Series (if they come in as DataFrames, take the first column)
    high_series = stock_data['High']
    if isinstance(high_series, pd.DataFrame):
        high_series = high_series.iloc[:, 0]
    
    ma_series = stock_data['MA']
    if isinstance(ma_series, pd.DataFrame):
        ma_series = ma_series.iloc[:, 0]
    
    low_series = stock_data['Low']
    if isinstance(low_series, pd.DataFrame):
        low_series = low_series.iloc[:, 0]

    # Calculate Bull Power and Bear Power using the computed MA
    stock_data['Bull Power'] = high_series - ma_series
    stock_data['Bear Power'] = low_series - ma_series
    
    return stock_data

# Function to flatten DataFrame columns if a MultiIndex is present
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# Streamlit UI
def main():
    st.title("Elder-Ray Index Calculator")
    
    # --- Common Inputs ---
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL):", "AAPL")
    data_mode = st.radio("Select Data Mode", ["Historical", "Real-Time"])
    
    # Customization for moving average settings
    ma_period = st.number_input("Enter Moving Average Period", min_value=1, value=14)
    ma_type = st.selectbox("Select Moving Average Type", ["EMA", "SMA"])
    price_column = st.selectbox("Select Price Column for Moving Average", ["Close", "Open", "High", "Low"], index=0)
    
    # --- Data Mode Specific Inputs ---
    if data_mode == "Historical":
        start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
        if start_date >= end_date:
            st.error("End Date must be after Start Date.")
            return
        
        if st.button("Calculate Elder-Ray Index (Historical)"):
            with st.spinner("Fetching historical data and calculating Elder-Ray Index..."):
                stock_data = fetch_stock_data(ticker, data_mode, start_date=start_date, end_date=end_date)
                if stock_data.empty:
                    st.error("No data found for the given stock symbol and date range.")
                    return
                elder_ray_index = calculate_elder_ray_index(stock_data, ma_period, ma_type, price_column)
                elder_ray_index = flatten_columns(elder_ray_index)  # Flatten columns if necessary
                
                st.subheader(f"Elder-Ray Index for {ticker} from {start_date} to {end_date}")
                st.write(elder_ray_index[["Bull Power", "Bear Power"]].tail())
                st.line_chart(elder_ray_index[["Bull Power", "Bear Power"]])
                
                # Option to show raw data
                if st.checkbox("Show Raw Data"):
                    st.dataframe(elder_ray_index)
                    
                # Download button for CSV export
                csv = elder_ray_index.to_csv().encode("utf-8")
                st.download_button("Download Data as CSV", data=csv, file_name=f"{ticker}_elder_ray_index.csv", mime="text/csv")
                
    else:  # Real-Time Data Mode
        period = st.text_input("Enter Period for Real-Time Data (e.g., 1d, 5d)", "1d")
        interval = st.text_input("Enter Interval (e.g., 1m, 5m, 15m)", "1m")
        
        if st.button("Calculate Elder-Ray Index (Real-Time)"):
            with st.spinner("Fetching real-time data and calculating Elder-Ray Index..."):
                stock_data = fetch_stock_data(ticker, data_mode, period=period, interval=interval)
                if stock_data.empty:
                    st.error("No data found for the given real-time settings.")
                    return
                elder_ray_index = calculate_elder_ray_index(stock_data, ma_period, ma_type, price_column)
                elder_ray_index = flatten_columns(elder_ray_index)  # Flatten columns if necessary
                
                st.subheader(f"Real-Time Elder-Ray Index for {ticker} (Period: {period}, Interval: {interval})")
                st.write(elder_ray_index[["Bull Power", "Bear Power"]].tail())
                st.line_chart(elder_ray_index[["Bull Power", "Bear Power"]])
                
                # Option to show raw data
                if st.checkbox("Show Raw Data"):
                    st.dataframe(elder_ray_index)
                    
                # Download button for CSV export
                csv = elder_ray_index.to_csv().encode("utf-8")
                st.download_button("Download Data as CSV", data=csv, file_name=f"{ticker}_real_time_elder_ray_index.csv", mime="text/csv")

if __name__ == "__main__":
    main()
