import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------------------
# Function to fetch stock data from Yahoo Finance
# -------------------------------------------
def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data for the given ticker symbol.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL').
    - period: Data period to fetch (e.g., '1y' for one year).
    - interval: Data interval (e.g., '1d' for daily data).
    
    Returns:
    - DataFrame containing the stock data.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            st.error("No data found. Please check the ticker symbol.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# -------------------------------------------
# Helper function to flatten MultiIndex columns (if present)
# -------------------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the DataFrame has MultiIndex columns, flatten them.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join([str(i) for i in col]).strip() for col in df.columns.values]
    return df

# -------------------------------------------
# Helper function to standardize column names and remove common trailing tokens
# -------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to lowercase, remove extra whitespace, and if all columns share a common trailing token,
    remove that trailing token.
    """
    # Convert to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()
    
    cols = df.columns.tolist()
    # Split each column name by whitespace
    split_cols = [col.split() for col in cols]
    
    # If all columns have at least two tokens, check for a common trailing token
    if all(len(tokens) >= 2 for tokens in split_cols):
        # Extract the last token from each column name
        last_tokens = [tokens[-1] for tokens in split_cols]
        # If all last tokens are the same, remove them from each column name
        if len(set(last_tokens)) == 1:
            new_cols = [' '.join(tokens[:-1]) for tokens in split_cols]
            df.columns = new_cols
    
    return df

# -------------------------------------------
# Function to calculate the Mass Index
# -------------------------------------------
def calculate_mass_index(data: pd.DataFrame, ema_period: int = 9, sum_period: int = 25) -> pd.Series:
    """
    Calculate the Mass Index indicator.
    
    The Mass Index is calculated in the following steps:
      1. Compute the daily range: (high - low)
      2. Calculate the exponential moving average (EMA) of the range using the specified EMA period.
      3. Calculate the EMA of the previously computed EMA.
      4. Compute the ratio: EMA(range) / EMA(EMA(range)).
      5. Calculate the Mass Index as the rolling sum of the ratio over the specified sum period.
    
    Parameters:
    - data: DataFrame containing at least 'high' and 'low' columns.
    - ema_period: Period for calculating the exponential moving averages (default is 9).
    - sum_period: Look-back period over which to sum the ratio (default is 25).
    
    Returns:
    - A Pandas Series representing the Mass Index.
    """
    # Flatten columns (if needed) and standardize the column names
    data = flatten_columns(data)
    data = standardize_columns(data)
    
    # Check if the required columns exist
    required_cols = {"high", "low"}
    if not required_cols.issubset(set(data.columns)):
        st.error(f"Data must contain the columns {required_cols}. Available columns: {list(data.columns)}")
        return pd.Series(dtype=float)
    
    # Calculate the daily price range
    price_range = data['high'] - data['low']
    
    # Calculate the first EMA of the range
    ema_range = price_range.ewm(span=ema_period, adjust=False).mean()
    
    # Calculate the EMA of the previous EMA (Double EMA)
    ema_ema_range = ema_range.ewm(span=ema_period, adjust=False).mean()
    
    # Compute the ratio of the two EMAs
    ratio = ema_range / ema_ema_range
    
    # Calculate the Mass Index as the rolling sum of the ratio over the sum_period
    mass_index = ratio.rolling(window=sum_period).sum()
    
    return mass_index

# -------------------------------------------
# Streamlit UI Code with Additional Customization Features
# -------------------------------------------
def main():
    st.title("Stock Data and Mass Index Calculator")
    
    # Sidebar for general configuration
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    period_str = st.sidebar.selectbox("Data Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=3)
    interval = st.sidebar.selectbox("Data Interval", options=["1d", "1wk", "1mo"], index=0)
    
    # Sidebar for Mass Index parameters
    st.sidebar.subheader("Mass Index Parameters")
    ema_period = st.sidebar.number_input("EMA Period", min_value=5, max_value=20, value=9, step=1)
    sum_period = st.sidebar.number_input("Sum Period", min_value=10, max_value=50, value=25, step=1)
    
    # Sidebar for Chart Customization
    st.sidebar.subheader("Chart Customization")
    line_color = st.sidebar.color_picker("Mass Index Line Color", value="#0000FF")
    line_style_choice = st.sidebar.selectbox("Mass Index Line Style", options=["solid", "dashed", "dotted", "dashdot"], index=0)
    # Map the chosen style to matplotlib line styles
    line_style_map = {
        "solid": "-",
        "dashed": "--",
        "dotted": ":",
        "dashdot": "-."
    }
    line_style = line_style_map[line_style_choice]
    
    chart_width = st.sidebar.slider("Chart Width (inches)", min_value=5, max_value=20, value=10)
    chart_height = st.sidebar.slider("Chart Height (inches)", min_value=3, max_value=15, value=4)
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    
    # Sidebar for Thresholds and Additional Options
    st.sidebar.subheader("Thresholds and Additional Options")
    show_thresholds = st.sidebar.checkbox("Show Threshold Lines", value=False)
    if show_thresholds:
        upper_threshold = st.sidebar.number_input("Upper Threshold", value=27.0)
        lower_threshold = st.sidebar.number_input("Lower Threshold", value=26.5)
    else:
        upper_threshold = None
        lower_threshold = None
    
    show_price_chart = st.sidebar.checkbox("Show Closing Price Chart", value=False)
    show_raw_table = st.sidebar.checkbox("Show Raw Stock Data Table", value=True)
    show_data_with_mi_table = st.sidebar.checkbox("Show Data with Mass Index Table", value=True)
    
    # Button to fetch data
    if st.sidebar.button("Fetch Data"):
        st.info(f"Fetching data for {ticker}...")
        stock_data = fetch_stock_data(ticker, period=period_str, interval=interval)
        if not stock_data.empty:
            st.subheader("Fetched Stock Data")
            if show_raw_table:
                st.dataframe(stock_data.tail(10))  # Display last 10 rows
            st.session_state['stock_data'] = stock_data
        else:
            st.error("Failed to fetch data. Please check the ticker symbol and parameters.")
    
    # Button to calculate and display Mass Index
    if st.sidebar.button("Calculate Mass Index"):
        if 'stock_data' not in st.session_state:
            st.error("Please fetch the stock data first.")
        else:
            stock_data = st.session_state['stock_data']
            st.info("Calculating Mass Index...")
            mass_index_series = calculate_mass_index(stock_data, ema_period=ema_period, sum_period=sum_period)
            
            # Append the Mass Index to the data for visualization
            stock_data_with_mi = stock_data.copy()
            stock_data_with_mi['Mass Index'] = mass_index_series
            
            # Plot the Mass Index Chart
            st.subheader("Mass Index Chart")
            fig, ax = plt.subplots(figsize=(chart_width, chart_height))
            ax.plot(stock_data_with_mi.index, stock_data_with_mi['Mass Index'],
                    label='Mass Index', color=line_color, linestyle=line_style)
            ax.set_title(f"{ticker} Mass Index (EMA Period: {ema_period}, Sum Period: {sum_period})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Mass Index")
            if show_grid:
                ax.grid(True)
            if show_thresholds and upper_threshold is not None and lower_threshold is not None:
                ax.axhline(upper_threshold, color='red', linestyle='--', label=f'Upper Threshold ({upper_threshold})')
                ax.axhline(lower_threshold, color='green', linestyle='--', label=f'Lower Threshold ({lower_threshold})')
            ax.legend()
            st.pyplot(fig)
            
            # Optionally plot the Closing Price Chart
            if show_price_chart:
                st.subheader("Closing Price Chart")
                fig2, ax2 = plt.subplots(figsize=(chart_width, chart_height))
                # Use the 'close' column from the fetched data
                if 'close' in stock_data.columns:
                    ax2.plot(stock_data.index, stock_data['close'], label='Close Price', color='orange')
                    ax2.set_title(f"{ticker} Closing Price")
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("Price")
                    if show_grid:
                        ax2.grid(True)
                    ax2.legend()
                    st.pyplot(fig2)
                else:
                    st.error("Closing price data not available.")
            
            # Show Data Tables if selected
            if show_data_with_mi_table:
                st.subheader("Data with Mass Index")
                st.dataframe(stock_data_with_mi.tail(10))
            
            # Provide a download button for the computed data as CSV
            csv_data = stock_data_with_mi.to_csv().encode('utf-8')
            st.download_button(label="Download Data as CSV",
                               data=csv_data,
                               file_name=f"{ticker}_mass_index.csv",
                               mime='text/csv')

if __name__ == "__main__":
    main()
