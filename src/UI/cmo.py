#!/usr/bin/env python3
import sys
import os
import pandas as pd
import streamlit as st
from yahooquery import Ticker
from dotenv import load_dotenv

# Load environment variables if needed (e.g., API keys)
load_dotenv()

# Adjust the system path so that our modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fetch_stock_data(ticker_symbol, period='1y'):
    """
    Fetch historical stock data for a given ticker symbol using yahooquery.
    Ensures that the DataFrame contains the required columns: date, high, low, and close.
    """
    st.info(f"Fetching historical data for {ticker_symbol} (period={period})...")
    ticker = Ticker(ticker_symbol)
    data = ticker.history(period=period)

    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)
    else:
        st.error("Failed to fetch data as a DataFrame.")
        return None

    # Ensure required columns exist; rename if necessary.
    for col in ['date', 'high', 'low', 'close']:
        if col not in data.columns:
            # Try converting from capitalized names if needed.
            if col.capitalize() in data.columns:
                data.rename(columns={col.capitalize(): col}, inplace=True)
            else:
                st.error(f"Required column '{col}' not found in data.")
                return None
    return data

class CMOCalculator:
    """
    Calculates the Chande Momentum Oscillator (CMO) based on price changes over a specified period.
    
    Default Calculation:
    - Gains (gain) = price_change if > 0 else 0
    - Losses (loss) = -price_change if < 0 else 0
    - Rolling sum over 'period' for gains and losses
    - CMO = 100 * (sum_of_gains - sum_of_losses) / (sum_of_gains + sum_of_losses)
    """
    def __init__(
        self,
        df,
        period=14,
        calc_method='Standard',
        apply_smoothing=None,
        smoothing_period=3,
        keep_intermediate=False
    ):
        """
        :param df: The input DataFrame (must include 'close').
        :param period: The lookback window for the CMO calculation.
        :param calc_method: Method for calculating gains/losses. Options: 'Standard', 'Absolute'.
        :param apply_smoothing: Smoothing method for final CMO. Options: None, 'SMA', 'EMA'.
        :param smoothing_period: Period used for smoothing the CMO values.
        :param keep_intermediate: If True, retains intermediate columns after calculation.
        """
        self.df = df.copy()
        self.period = period
        self.calc_method = calc_method
        self.apply_smoothing = apply_smoothing
        self.smoothing_period = smoothing_period
        self.keep_intermediate = keep_intermediate

    def calculate(self):
        # Calculate price changes
        self.df['price_change'] = self.df['close'].diff()

        # Option 1: Standard Gains/Losses
        # Option 2: Absolute Gains/Losses (treat all changes as gains but separate out positivity in a second step)
        if self.calc_method == 'Absolute':
            # Gains are the absolute value of price_change
            self.df['gain'] = self.df['price_change'].abs()
            # Loss is 0 in this approach (or you could interpret differently)
            self.df['loss'] = 0
        else:
            # Standard approach
            self.df['gain'] = self.df['price_change'].where(self.df['price_change'] > 0, 0)
            self.df['loss'] = -self.df['price_change'].where(self.df['price_change'] < 0, 0)

        # Calculate the rolling sums of gains and losses over the specified period
        self.df['gain_sum'] = self.df['gain'].rolling(window=self.period).sum()
        self.df['loss_sum'] = self.df['loss'].rolling(window=self.period).sum()

        # Calculate the CMO
        self.df['cmo'] = 100 * (self.df['gain_sum'] - self.df['loss_sum']) / (self.df['gain_sum'] + self.df['loss_sum'])

        # Apply optional smoothing on the CMO (SMA or EMA)
        if self.apply_smoothing == 'SMA':
            self.df['cmo'] = self.df['cmo'].rolling(window=self.smoothing_period).mean()
        elif self.apply_smoothing == 'EMA':
            self.df['cmo'] = self.df['cmo'].ewm(span=self.smoothing_period, adjust=False).mean()

        # Drop intermediate columns if the user doesn't want them
        if not self.keep_intermediate:
            self.df.drop(columns=['price_change', 'gain', 'loss', 'gain_sum', 'loss_sum'], inplace=True)

        return self.df

def highlight_cmo_above_threshold(val, threshold):
    """Highlight cells if CMO goes above threshold (simple example styling)."""
    color = 'red' if val >= threshold else 'black'
    return f'color: {color}'

def main():
    st.title("Chande Momentum Oscillator (CMO) Calculation System")
    st.write("Customize and calculate the CMO for your selected stock.")

    # Input for stock symbol and data period.
    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    period_option = st.selectbox("Select Data Period:", options=["1y", "6mo", "3mo", "1mo"], index=0)

    # -- CMO Customization Options --
    st.subheader("CMO Parameters")
    # 1. Base period
    cmo_period = st.number_input("CMO Calculation Period:", min_value=1, max_value=200, value=14)

    # 2. Gains/Losses Method
    calc_method = st.selectbox(
        "Gains/Loss Calculation Method:",
        options=["Standard", "Absolute"],
        index=0
    )

    # 3. Smoothing
    apply_smoothing = st.selectbox(
        "Apply Smoothing to CMO?",
        options=[None, "SMA", "EMA"],
        format_func=lambda x: "None" if x is None else x,
        index=0
    )
    smoothing_period = st.number_input("Smoothing Period (for SMA/EMA):", min_value=1, max_value=50, value=3)

    # 4. Keep or drop intermediate columns
    keep_intermediate = st.checkbox("Keep intermediate columns for debugging?", value=False)

    # 5. Threshold highlight
    threshold_enable = st.checkbox("Enable threshold highlight on CMO?", value=False)
    threshold_value = None
    if threshold_enable:
        threshold_value = st.number_input("Highlight CMO above threshold value:", value=70)

    # Button to calculate and display CMO data.
    if st.button("Calculate CMO"):
        # Fetch the historical data.
        data = fetch_stock_data(ticker_symbol, period=period_option)
        if data is not None:
            st.subheader(f"Original Stock Data for {ticker_symbol}")
            st.dataframe(data.tail(10))

            # Calculate the CMO
            cmo_calc = CMOCalculator(
                data,
                period=cmo_period,
                calc_method=calc_method,
                apply_smoothing=apply_smoothing,
                smoothing_period=smoothing_period,
                keep_intermediate=keep_intermediate
            )
            cmo_data = cmo_calc.calculate()

            st.subheader("Calculated CMO Data")
            if threshold_enable and threshold_value is not None:
                # Use a style function to highlight values above threshold.
                styled_cmo_data = cmo_data.style.applymap(
                    lambda val: highlight_cmo_above_threshold(val, threshold_value),
                    subset=['cmo']  # Only apply styling to the CMO column
                )
                st.dataframe(styled_cmo_data)
            else:
                st.dataframe(cmo_data.tail(20))

    # Button to fetch and display the latest data without calculating CMO.
    if st.button("Fetch Latest Data"):
        latest_data = fetch_stock_data(ticker_symbol, period=period_option)
        if latest_data is not None:
            st.subheader(f"Latest Stock Data for {ticker_symbol}")
            st.dataframe(latest_data.tail(10))

if __name__ == '__main__':
    main()
