import yfinance_cache as yfc
import pandas as pd
import streamlit as st
import sys
import time
from datetime import date, datetime, timedelta
import requests_cache
import concurrent.futures
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

progress_bar = None
progress_counter = 0
stock_tickers_amount = 0
currency_rates = {} # Cache for currency rates

# --- Helper Functions (Modified to accept dataframes/info dicts) ---

# Function to calculate w13612 momentum
def calculate_13612W(history_1y):
    if history_1y is None or history_1y.empty:
        return -100
    try:
        dfmonthly = history_1y.groupby([pd.Grouper(freq='ME')]).last()
        if len(dfmonthly) < 12: # Need at least 12 months for the longest shift
             return -100

        # Ensure required columns exist after grouping
        if 'Close' not in dfmonthly.columns:
            return -100

        # Calculate shifts safely
        shift1 = dfmonthly['Close'].shift(1)
        shift3 = dfmonthly['Close'].shift(3)
        shift6 = dfmonthly['Close'].shift(6)
        shift12 = dfmonthly['Close'].shift(12)

        # Calculate momentum components, handling potential division by zero or NaN
        mom1 = (dfmonthly['Close'] / shift1 - 1).fillna(0) * 12
        mom3 = (dfmonthly['Close'] / shift3 - 1).fillna(0) * 4
        mom6 = (dfmonthly['Close'] / shift6 - 1).fillna(0) * 2
        mom12 = (dfmonthly['Close'] / shift12 - 1).fillna(0) * 1

        w13612 = (mom1 + mom3 + mom6 + mom12) / 4

        return w13612.iloc[-1] if not w13612.empty else -100
    except Exception as e:
        # print(f"Error calculating 13612W: {e}")
        return -100


# Function to calculate 5y total return rate in $
def calculate_5y_total_return_rate(info, history_5y, dividends_5y, currency_rates_local):
    # ticker_symbol = info.get('symbol', 'N/A') # Get ticker for logging - Removed debug print

    if history_5y is None or history_5y.empty or dividends_5y is None:
        # print(f"[{ticker_symbol}] Returning 0 due to missing history or dividends.") # DEBUG Removed
        return 0

    # Ensure current_time is timezone-naive for comparisons
    current_time = datetime.now()
    fiveYearsAgo = current_time - timedelta(days=5*365 + 2) # Approximate start date (also naive)

    try:
        # --- Timezone Handling for Start Date ---
        # Work with a timezone-naive version of the history index for comparison
        history_index_naive = history_5y.index
        if hasattr(history_index_naive, 'tz') and history_index_naive.tz is not None:
            history_index_naive = history_index_naive.tz_localize(None)

        # Find the closest available naive date in history to 5 years ago
        actual_start_date_naive = history_index_naive[history_index_naive >= fiveYearsAgo].min()

        if pd.isna(actual_start_date_naive):
             # print(f"[{ticker_symbol}] Could not find history data close to 5 years ago (target: {fiveYearsAgo}).") # DEBUG Removed
             return 0 # Not enough history

        # Get the original (potentially timezone-aware) start date using the naive version to locate it
        actual_start_date = history_5y.index[history_index_naive == actual_start_date_naive][0]
        # --- End Timezone Handling ---

        stock_price_5y_ago = history_5y.loc[actual_start_date]['Close']
        current_price = info.get('previousClose', history_5y.iloc[-1]['Close']) # Use previousClose if available

        # print(f"[{ticker_symbol}] Start Date: {actual_start_date}, Price 5y ago: {stock_price_5y_ago}, Current Price: {current_price}") # DEBUG Removed

        if current_price is None or stock_price_5y_ago is None or stock_price_5y_ago == 0:
            # print(f"[{ticker_symbol}] Returning 0 due to missing prices or zero start price.") # DEBUG Removed
            return 0

        initialRate = 1.0
        endingRate = 1.0
        currency_code = info.get('currency', 'USD')

        if currency_code != 'USD':
            rate_info = currency_rates_local.get(currency_code)
            if rate_info:
                initialRate = rate_info.get('initial', 1.0)
                endingRate = rate_info.get('ending', 1.0)
                # print(f"[{ticker_symbol}] Currency {currency_code}: Initial Rate={initialRate}, Ending Rate={endingRate}") # DEBUG Removed
            else:
                # print(f"[{ticker_symbol}] Warning: Missing currency rate for {currency_code}. Using 1.0.") # DEBUG Removed
                pass

        # Handle GBp (pence) - assuming prices are in pence if currency is GBp
        if currency_code == 'GBp':
             # print(f"[{ticker_symbol}] Adjusting GBp prices.") # DEBUG Removed
             stock_price_5y_ago /= 100.0
             current_price /= 100.0
             # Check if dividends are also in pence - yfinance usually provides them in base currency units (GBP)
             # If dividends were in pence, they'd need division too. Assuming they are in GBP.

        # --- Timezone Handling for Dividend Slicing ---
        # Work with a timezone-naive version of the dividends index for slicing
        dividends_naive = dividends_5y.copy() # Avoid modifying original
        if hasattr(dividends_naive.index, 'tz') and dividends_naive.index.tz is not None:
            dividends_naive.index = dividends_naive.index.tz_localize(None)

        # Slice using the naive dates
        dividends_period = dividends_naive.loc[actual_start_date_naive:current_time]
        totalDividend = dividends_period.sum() # Correct: Sum the Series directly
        # --- End Timezone Handling ---
        # print(f"[{ticker_symbol}] Total dividends in period ({actual_start_date_naive} to {current_time}): {totalDividend}") # DEBUG Removed


        # Apply average exchange rate to dividends if not USD
        avgRate = (endingRate + initialRate) / 2
        totalDividend_adj = totalDividend * avgRate if currency_code != 'USD' else totalDividend
        # print(f"[{ticker_symbol}] Adjusted Total Dividends (Avg Rate {avgRate}): {totalDividend_adj}") # DEBUG Removed


        # Calculate return components
        dividend_return = totalDividend_adj * 0.6 # Assuming 40% tax
        capital_gain = (endingRate * current_price) - (initialRate * stock_price_5y_ago)
        initial_value_adj = initialRate * stock_price_5y_ago

        # print(f"[{ticker_symbol}] Dividend Return (60%): {dividend_return}, Capital Gain: {capital_gain}, Initial Value Adj: {initial_value_adj}") # DEBUG Removed


        if initial_value_adj == 0:
            # print(f"[{ticker_symbol}] Returning 0 due to zero initial adjusted value.") # DEBUG Removed
            return 0

        total_return = (dividend_return + capital_gain) / initial_value_adj
        # print(f"[{ticker_symbol}] Calculated Total Return: {total_return * 100}%") # DEBUG Removed
        return total_return * 100 # Return as percentage

    except Exception as e:
        # print(f"[{ticker_symbol}] Error calculating 5y return: {e}") # DEBUG Removed
        # Optionally log the error properly
        # import traceback
        # print(f"Error calculating 5y return for {info.get('symbol', 'N/A')}: {e}\n{traceback.format_exc()}")
        return 0


# Function to calculate dividend growth rate
def calculate_dividend_growth_rate(dividends_all):
    if dividends_all is None or dividends_all.empty:
        return 0
    try:
        # Ensure index is datetime
        dividends_all.index = pd.to_datetime(dividends_all.index)
        resampled = dividends_all.resample('YE').sum()

        if len(resampled) < 6: # Need at least 6 years for a 5-year growth calculation
            return 0

        # Determine the index for the end year (idx) and start year (idx-5)
        # If the last year is the current year and potentially incomplete, use the previous year as the end year
        last_year_end_date = resampled.index[-1]
        if last_year_end_date.year == date.today().year and last_year_end_date.dayofyear < 365:
             if len(resampled) < 7: # Need 7 years if skipping the last partial year
                 return 0
             idx = -2 # Use year before last as end year
        else:
             idx = -1 # Use last full year as end year

        end_dividend = resampled.iloc[idx]
        start_dividend = resampled.iloc[idx - 5]

        if start_dividend <= 0: # Avoid division by zero or growth from zero
            return 0

        # CAGR formula: ((Ending Value / Starting Value)^(1 / Number of Years)) - 1
        growth_rate = ((end_dividend / start_dividend) ** (1 / 5)) - 1
        return growth_rate * 100 # Return as percentage

    except Exception as e:
        # print(f"Error calculating dividend growth: {e}")
        return 0


# Function to filter bad dividends data in Yahoo Finance
def filter_wrong_dividends(dividends_df):
    if dividends_df is None or dividends_df.empty:
        return dividends_df

    # Ensure the index is sorted
    dividends_df = dividends_df.sort_index()

    # Use shift to compare consecutive rows efficiently
    prev_value = dividends_df['Dividends'].shift(1)
    prev_index = pd.Series(dividends_df.index, index=dividends_df.index).shift(1)

    # Conditions for filtering:
    # 1. Same dividend value as the previous row
    # 2. Time difference is less than 2 days
    is_duplicate = (dividends_df['Dividends'] == prev_value) & \
                   ((dividends_df.index - prev_index) < timedelta(days=2))

    # Keep rows that are NOT duplicates
    filtered_df = dividends_df[~is_duplicate]

    return filtered_df


# Function to calculate dividend streak
def calculate_dividend_streak(dividends_all):
    if dividends_all is None or dividends_all.empty:
        return 0

    dividends_df = dividends_all.to_frame(name='Dividends')
    filtered_dividends_df = filter_wrong_dividends(dividends_df)

    if filtered_dividends_df.empty:
        return 0

    try:
        # Ensure index is datetime
        filtered_dividends_df.index = pd.to_datetime(filtered_dividends_df.index)
        resampled = filtered_dividends_df['Dividends'].resample('YE').sum()

        # Check if the last dividend was more than ~1 year ago from today
        if resampled.empty or (date.today() - resampled.index[-1].date() > timedelta(days=380)):
             return 0

        current_streak = 0
        last_valid_year_dividend = -1 # Track the dividend amount of the last year in the current streak

        for year_end_date in resampled.index:
            current_year_dividend = resampled[year_end_date]

            # Check for growth or stability compared to the last valid year in the streak
            if current_year_dividend >= last_valid_year_dividend and current_year_dividend > 0:
                 # If it's the first year or consecutive year, increment streak
                 current_streak += 1
                 last_valid_year_dividend = current_year_dividend # Update last valid dividend amount
            else:
                 if year_end_date != resampled.index[-1]: # ignore last year in the series as it may be incomplete
                    # Streak broken, reset, but check if the current year starts a new streak
                    if current_year_dividend > 0:
                        current_streak = 1
                        last_valid_year_dividend = current_year_dividend
                    else:
                        current_streak = 0
                        last_valid_year_dividend = -1 # Reset since dividend is zero or negative

        return current_streak # Final check for the ongoing streak

    except Exception as e:
        # print(f"Error calculating dividend streak: {e}")
        return 0


# --- Main Data Fetching ---

# Function to fetch stock data for a single ticker
def fetch_stock_data_single(ticker):
    print(f"Fetching {ticker}")
    try:
        stock = yfc.Ticker(ticker)
        info = stock.info

        # Fetch history needed for all calculations (e.g., 6 years for 5y growth + buffer)
        history_6y = stock.history(period="6y")
        dividends_all = stock._dat.dividends # Fetch all dividends once (Use public attribute)

        if history_6y is None or history_6y.empty:
            print(f"No history for {ticker}")
            # Decide how to handle: return None or partial data? Returning None for now.
            return None # Corrected indentation

        # Slice history for specific calculations
        # Calculate the date one year ago from the latest date in the index
        one_year_ago = history_6y.index.max() - pd.DateOffset(years=1) # Corrected indentation
        history_1y = history_6y.loc[history_6y.index >= one_year_ago] # Corrected indentation
        history_5y = history_6y # Use the 6y fetch for 5y calculations # Corrected indentation

        # Pre-calculate values from info
        fiveYearAvgDividendYield = info.get('fiveYearAvgDividendYield') # Corrected indentation
        dividendRate = info.get('dividendRate') # Corrected indentation
        close = info.get('previousClose', history_6y.iloc[-1]['Close'] if not history_6y.empty else 0) # Fallback to last close # Corrected indentation
        currency = info.get('currency', 'USD') # Corrected indentation

        discount = 0 # Corrected indentation
        if close and fiveYearAvgDividendYield and dividendRate: # Corrected indentation
            try: # Corrected indentation
                # Adjust calculation for GBp (pence) if necessary
                # Assuming dividendRate is in base currency (GBP), fiveYearAvgDividendYield is percentage
                if currency == 'GBp': # Corrected indentation
                    # Convert close from pence to pounds for calculation consistency
                    close_pounds = close / 100.0 # Corrected indentation
                    if close_pounds > 0 and fiveYearAvgDividendYield > 0: # Corrected indentation
                         # dividendRate is likely annual GBP, yield is %, need yield as decimal
                         intrinsic_value = dividendRate / (fiveYearAvgDividendYield / 100.0) # Corrected indentation
                         discount = 100 * (close_pounds - intrinsic_value) / close_pounds # Corrected indentation
                else: # Corrected indentation
                    if close > 0 and fiveYearAvgDividendYield > 0: # Corrected indentation
                         intrinsic_value = dividendRate / (fiveYearAvgDividendYield / 100.0) # Corrected indentation
                         discount = 100 * (close - intrinsic_value) / close # Corrected indentation
            except ZeroDivisionError: # Corrected indentation
                discount = 0 # Avoid division by zero if yield or close is zero # Corrected indentation
            except Exception as e: # Corrected indentation
                # print(f"Error calculating discount for {ticker}: {e}")
                discount = 0 # Corrected indentation


        # Pass fetched data to helper functions
        data = { # Corrected indentation
            'ticker': f"https://finance.yahoo.com/quote/{ticker}/",
            'name': info.get('shortName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'country': info.get('country', ''),
            'currency': currency, # Store currency for potential later use
            'dividend_streak': calculate_dividend_streak(dividends_all),
            'dividend_yield': round(info.get('dividendYield', 0), 2),
            '5y_Avg_dividend_yield': round(fiveYearAvgDividendYield if fiveYearAvgDividendYield else 0, 2),
            'payout_ratio': round(info.get('payoutRatio', 0), 2),
            'dividend_growth_rate': round(calculate_dividend_growth_rate(dividends_all), 2),
            'eps': round(info.get('trailingEps', 0), 2),
            'pe_ratio': round(info.get('trailingPE', 0), 2),
            'earning_growth': info.get('earningsGrowth', 0),
            # Pass currency_rates (global) to 5y calculation
            '5y_total_return': round(calculate_5y_total_return_rate(info, history_5y, dividends_all, currency_rates), 2),
            'debt_to_equity': round(info.get('debtToEquity', 0), 2),
            'roe': round(info.get('returnOnEquity', 0), 2),
            'discount/premium': round(discount, 2),
            'w13612': round(calculate_13612W(history_1y), 2),
        }
        return data # Corrected indentation

    except Exception as e: # Corrected indentation
        print(f"Exception processing {ticker}: {e}") # Corrected indentation
        return None
    finally:
        # Update progress bar from the worker thread (needs thread-safe update)
        # This part is tricky with Streamlit's progress bar from threads.
        # A queue or callback mechanism might be needed for perfect accuracy.
        # For simplicity, we'll update based on completed futures later.
        pass


# --- Streamlit UI and Filtering ---

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df


# --- Data Loading and Processing ---

def convert_symbol_to_yfinance(symbol, country):
    match country:
        case 'AE':
            return symbol.upper() + '.AE'
        case 'AU':
            return symbol.upper() + '.AX'
        case 'BE':
            return symbol.upper() + '.BR'
        case 'BM':
            return symbol.upper()
        case 'BR':
            return symbol.upper() + '.SA'
        case 'CA':
            return symbol.upper() + '.TO'
        case 'CH':
            return symbol.upper() + '.SW'
        case 'CN':
            return symbol.upper() + '.SS'
        case 'DE':
            return symbol.upper() + '.DE'
        case 'DK':
            return symbol.upper() + '.CO'
        case 'ES':
            return symbol.upper() + '.MC'
        case 'FI':
            return symbol.upper() + '.HE'
        case 'FR':
            return symbol.upper() + '.PA'
        case 'GB':
            return symbol.upper() + '.L'
        case 'HK':
            return symbol.upper() + '.HK'
        case 'ID':
            return symbol.upper() + '.JK'
        case 'IN':
            return symbol.upper() + '.NS'
        case 'IT':
            return symbol.upper() + '.MI'
        case 'JP':
            return symbol.upper() + '.T'
        case 'KR':
            return symbol.upper() + '.KS'
        case 'KW':
            return symbol.upper() + '.KW'
        case 'MX':
            return symbol.upper() + '.MX'
        case 'MY':
            return symbol.upper() + '.KL'
        case 'NL':
            return symbol.upper() + '.AS'
        case 'NO':
            return symbol.upper() + '.OL'
        case 'NZ':
            return symbol.upper() + '.AX'
        case 'PR':
            return symbol.upper()
        case 'QA':
            return symbol.upper() + '.QA'
        case 'RU':
            return symbol.upper() + '.RU'
        case 'SA':
            return symbol.upper() + '.SR'
        case 'SE':
            return symbol.upper() + '.ST'
        case 'SG':
            return symbol.upper() + '.SI'
        case 'TH':
            return symbol.upper() + '.BK'
        case 'TW':
            return symbol.upper() + '.TW'
        case 'US':
            return symbol.upper()
        case 'ZA':
            return symbol.upper() + '.JO'
        case _:
            raise ValueError(f"Unknown country {country}")


# Read stock tickers from file
def read_stock_tickers(file_path):
    """Reads tickers from a file. Handles CSV (with 'Symbol' or 'BBG FIGI') or plain text."""
    tickers = []
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, header=0)

            tickers = [convert_symbol_to_yfinance(symbol, country) 
                        for symbol, country in zip(df['Symbol'], df['Country'])]

        else:
            # If not a CSV, read it line by line (plain text ticker list)
            with open(file_path, 'r') as file:
                tickers = [line.strip() for line in file if line.strip()]

        return tickers

    except FileNotFoundError:
        st.error(f"Error: Ticker file not found at {file_path}")
        return []
    except Exception as e:
        st.error(f"Error reading ticker file: {e}")
        return []

# Pre-fetch currency rates
def fetch_currency_rates(currencies_needed):
    global currency_rates
    print(f"Fetching currency rates for: {currencies_needed}")
    rates = {}
    current_time = datetime.now()
    fiveYearsAgo = current_time - timedelta(days=5*365 + 2)
    endDate = fiveYearsAgo + timedelta(days=5) # Fetch a small window

    for currency_code in currencies_needed:
        if currency_code == 'USD': continue # No conversion needed for USD

        ticker_symbol = f"{currency_code}USD=X"
        if currency_code == 'ILA': ticker_symbol = 'ILSUSD=X' # Handle specific codes if needed
        if currency_code == 'ZAc': ticker_symbol = 'ZARUSD=X'
        if currency_code == 'KWF': ticker_symbol = 'KWDUSD=X'
        # Add other mappings if necessary

        try:
            currency_ticker = yfc.Ticker(ticker_symbol)
            # Fetch current rate
            endingRate = currency_ticker.info.get('previousClose')
            if not endingRate: # Fallback if previousClose is missing
                 hist_now = currency_ticker.history(period="1d")
                 if not hist_now.empty:
                     endingRate = hist_now.iloc[-1]['Close']

            # Fetch rate 5 years ago
            initialRate = None
            hist_5y = currency_ticker.history(start=fiveYearsAgo.strftime("%Y-%m-%d"), end=endDate.strftime("%Y-%m-%d"), interval="1d")
            if not hist_5y.empty:
                 initialRate = hist_5y.iloc[0]['Close']

            if endingRate and initialRate:
                 rates[currency_code] = {'initial': initialRate, 'ending': endingRate}
                 # print(f"Fetched rates for {currency_code}: Initial={initialRate}, Ending={endingRate}")
            else:
                 print(f"Warning: Could not fetch full rate data for {currency_code}")
                 rates[currency_code] = {'initial': 1.0, 'ending': 1.0} # Fallback

        except Exception as e:
            print(f"Error fetching currency rate for {ticker_symbol}: {e}")
            rates[currency_code] = {'initial': 1.0, 'ending': 1.0} # Fallback on error

    currency_rates = rates # Store globally


# Calculate positive and negative metrics
def calculate_metrics(row, spy_return):

    positives = sum([
        row['dividend_streak'] > 10,
        row['dividend_yield'] > 0,
        0.30 <= row['payout_ratio'] <= 0.60,
        row['dividend_growth_rate'] > 10,
        row['eps'] > 0,
        row['pe_ratio'] < 20,
        row['debt_to_equity'] < 100,
        row['roe'] > 0.10,
        row['discount/premium'] < 0,
        row.get('5y_total_return', 0) > spy_return if spy_return is not None else False, # Handle case where spy_return is None
        row.get('earning_growth', 0) > 0,
        row.get('w13612', -100) > 0 # Use default if missing
    ])
    return positives

# Highlight positive and negative metrics
def highlight_metrics(row, df, spy_return):
    colors = []
    for col in df.columns:
        if col == 'dividend_streak':
            colors.append('background-color: green' if row[col] >= 10 else 'background-color: red')
        elif col == 'dividend_yield':
            colors.append('background-color: green' if row[col] > 0 else 'background-color: red')
        elif col == 'payout_ratio':
            colors.append('background-color: green' if 0.30 <= row[col] <= 0.60 else 'background-color: red')
        elif col == 'dividend_growth_rate':
            colors.append('background-color: green' if row[col] > 10 else 'background-color: red')
        elif col == 'eps':
            colors.append('background-color: green' if row[col] > 0 else 'background-color: red')
        elif col == 'pe_ratio':
            colors.append('background-color: green' if row[col] < 20 else 'background-color: red')
        elif col == 'debt_to_equity':
            colors.append('background-color: green' if row[col] < 100 else 'background-color: red')
        elif col == 'roe':
            colors.append('background-color: green' if row[col] > 0.10 else 'background-color: red')
        elif col == 'discount/premium':
            colors.append('background-color: green' if row[col] < 0 else 'background-color: red')
        elif col == '50d-SMA':
            colors.append('background-color: green' if row[col] > 0 else 'background-color: red')
        elif col == '200d-SMA':
            colors.append('background-color: green' if row[col] > 0 else 'background-color: red')
        elif col == 'w13612':
            colors.append('background-color: green' if row[col] > 0 else 'background-color: red')
        elif col == '5y_total_return':
            # Handle potential missing columns gracefully
            colors.append('background-color: green' if row.get(col, 0) > spy_return else 'background-color: red')
        elif col == 'earning_growth':
            colors.append('background-color: green' if row.get(col, 0) > 0 else 'background-color: red')
        else:
            colors.append('')
    return colors


# --- Main Loading Function ---

@st.cache_data(ttl=3600*24, show_spinner=False)
def load_data(ticker_file):
    global progress_bar, progress_counter, stock_tickers_amount, currency_rates

    progress_bar = st.progress(0, text="Reading tickers...")
    stock_tickers = read_stock_tickers(ticker_file)
    stock_tickers_amount = len(stock_tickers)
    if stock_tickers_amount == 0:
        st.warning("No tickers found in the file.")
        progress_bar.empty()
        return (pd.DataFrame(), None) # Return empty DataFrame and None for spy_return

    progress_counter = 0

    # --- Pre-fetch currency data ---
    # Need to know which currencies are involved. This requires a quick initial scan or assumptions.
    # For simplicity, let's assume we might need common ones, or fetch them dynamically later.
    # A better approach would be a preliminary fetch of just info['currency'] for all tickers.
    # Quick pre-fetch (less efficient but simpler for now):
    progress_bar.progress(0, text="Fetching currency info...")
    currencies_to_fetch = set()
    temp_infos = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(lambda t: (t, yfc.Ticker(t).info), ticker): ticker for ticker in stock_tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                t, info = future.result()
                if info and 'currency' in info and info['currency'] != 'USD':
                    currencies_to_fetch.add(info['currency'])
                temp_infos[ticker] = info # Store info temporarily if needed later
            except Exception as exc:
                print(f'{ticker} generated an exception during currency check: {exc}')
    fetch_currency_rates(list(currencies_to_fetch))
    # --- End Currency Pre-fetch ---


    progress_bar.progress(0, text="Fetching stock data...")
    stock_data = []
    # Use ThreadPoolExecutor for parallel fetching
    # Adjust max_workers based on testing to avoid rate limits (start lower)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_ticker = {executor.submit(fetch_stock_data_single, ticker): ticker for ticker in stock_tickers}

        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if data: # Only append if data was successfully fetched
                    stock_data.append(data)
            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')
            finally:
                # Update progress bar (thread-safe update needed for perfect accuracy)
                progress_counter += 1
                percent_complete = progress_counter / stock_tickers_amount
                progress_bar.progress(percent_complete, text=f'Processed {progress_counter}/{stock_tickers_amount} tickers...')
                # Optional small delay to potentially ease rate limiting
                # time.sleep(0.05)


    progress_bar.progress(1.0, text="Processing data...")
    if not stock_data:
        st.warning("No data fetched successfully for any ticker.")
        progress_bar.empty()
        return (pd.DataFrame(), None)

    stock_df = pd.DataFrame(stock_data)

    # Fetch SPY return separately (can also be parallelized if slow)
    spy_return = None
    try:
        print("Fetching SPY data for benchmark...")
        spy_ticker = yfc.Ticker('SPY')
        spy_info = spy_ticker.info
        spy_hist = spy_ticker.history(period="6y") # Match history period
        spy_divs = spy_ticker._dat.dividends # Use public attribute
        # SPY is USD, so currency_rates isn't strictly needed, pass empty dict or handle inside
        spy_return = calculate_5y_total_return_rate(spy_info, spy_hist, spy_divs, {})
        print(f"SPY 5y Return: {spy_return}")
    except Exception as e:
        print(f"Could not calculate SPY return: {e}")
        st.warning("Could not calculate SPY 5-year return for comparison.")


    # Calculate Score - ensure 'Score' column exists even if SPY fails
    if 'Score' not in stock_df.columns:
         stock_df['Score'] = 0

    if spy_return is not None:
        stock_df['Score'] = stock_df.apply(lambda row: calculate_metrics(row, spy_return), axis=1)
    else:
        # If SPY failed, maybe calculate score without the SPY comparison?
        # Or leave score as 0/NaN? Leaving as calculated without SPY comparison for now.
        temp_cols = stock_df.columns.drop('Score', errors='ignore') # Exclude score itself
        stock_df['Score'] = stock_df.apply(lambda row: calculate_metrics(row, None), axis=1)


    # Final processing
    # Drop rows where essential data might be missing if desired (e.g., ticker, name)
    # sorted_df = stock_df.dropna(subset=['ticker', 'name']) # Example
    sorted_df = stock_df.round(2)  # Round to two decimal places

    progress_bar.empty()
    print("Data loading complete.")
    return (sorted_df, spy_return)


# --- Main Execution ---

def main():
    st.set_page_config(page_title="Stock Screener", layout="wide")
    st.title("Stock Screener")

    # Get ticker file from command line argument
    if len(sys.argv) < 2:
        st.error("Please provide the ticker file path as a command line argument.")
        st.stop()
    ticker_file = sys.argv[1]

    # Load data using the cached function
    (sorted_df, spy_return) = load_data(ticker_file)

    if sorted_df.empty:
        st.warning("No stock data to display.")
        st.stop()

    # Display filtering options and filtered dataframe
    df_filtered = filter_dataframe(sorted_df)

    # Apply styling
    df_styled = df_filtered.style.apply(highlight_metrics, axis=1, args=(df_filtered, spy_return))

    # Display the styled dataframe
    st.dataframe(
        df_styled,
        use_container_width = True, 
        column_config = {"ticker": st.column_config.LinkColumn(
                "Ticker", max_chars=100, display_text=r"https://finance.yahoo.com/quote/(.*?)/")
                }, 
         hide_index=False)



if __name__ == "__main__":
    main()
