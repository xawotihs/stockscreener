import yfinance_cache as yfc
import pandas as pd
import streamlit as st
import sys
from datetime import date, datetime, timedelta
import requests_cache
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

progress_bar = None
stock_tickers_amount = 0


# Function to fetch stock data
def fetch_stock_data(ticker, percent_complete):
    progress_bar.progress(percent_complete, text=f'Fetching data for {ticker}')

    stock = yfc.Ticker(ticker)
    info = stock.info

    fiveYearAvgDividendYield = stock.info.get('fiveYearAvgDividendYield')
    dividendRate = stock.info.get('dividendRate')
    close = stock.info.get('previousClose')
    fiftyDayAverage = info.get('fiftyDayAverage', '')
    twoHundredDayAverage = info.get('twoHundredDayAverage', '')
    if close is None or close == 'N/A':
        close = 1
    if fiveYearAvgDividendYield is None or dividendRate is None or fiveYearAvgDividendYield == 'N/A':
        fiveYearAvgDividendYield = 0
        discount = 0
    else:
        if stock.info.get('currency') != 'GBp':
            discount = 100*(close-dividendRate/fiveYearAvgDividendYield*100)/(close)
        else:
            discount = 100*(close-(dividendRate*100)/fiveYearAvgDividendYield*100)/(close)

    try:
        return {
            'ticker': ticker,
            'name': info.get('shortName', ''),
            'sector': info.get('sector', ''),
            'country': info.get('country', ''),
            'dividend_streak': calculate_dividend_streak(stock),
            'dividend_yield': round(info.get('dividendYield', 0) * 100, 2),
            '5y_Avg_dividend_yield': round(fiveYearAvgDividendYield, 2),
            'payout_ratio': round(info.get('payoutRatio', 0), 2),
            'dividend_growth_rate': round(calculate_dividend_growth_rate(stock), 2),
            'eps': round(info.get('trailingEps', 0), 2),
            'pe_ratio': round(info.get('trailingPE', 0), 2),
            'earning_growth': info.get('earningsGrowth', 0),
            '5y_total_return': round(calculate_5y_total_return_rate(stock), 2),
            'debt_to_equity': round(info.get('debtToEquity', 0), 2),
            'roe': round(info.get('returnOnEquity', 0), 2),
            'discount/premium':round(discount,2),
            'w13612': round(calculate_13612W(stock), 2),
        }
    except KeyError:
        return None

# Function to calculate w13612 momentum
def calculate_13612W(stock):
    current_time = datetime.now()
    until = current_time - timedelta(days = 365+2)

    try:
        df = stock.history(start=until.strftime("%Y-%m-%d"), end=current_time.strftime("%Y-%m-%d"), interval="1d")
        dfmonthly = df.groupby([pd.Grouper(freq = 'ME')]).last()

        w13612 = ((dfmonthly['Close']/dfmonthly['Close'].shift(1)-1)*12+
                    (dfmonthly['Close']/dfmonthly['Close'].shift(3)-1)*4+
                    (dfmonthly['Close']/dfmonthly['Close'].shift(6)-1)*2+
                    (dfmonthly['Close']/dfmonthly['Close'].shift(12)-1))/4

        return w13612.iloc[-1]
    except Exception:
        return -100


# Function to calculate 5y total return rate in $
def calculate_5y_total_return_rate(stock):
    current_time = datetime.now()
    ctr = 5
    initialRate = 1
    endingRate = 1
    # assign old price to current price, just to be safe with failing APIs
    stock_price_5y_ago = stock.info['previousClose']
    while ctr > 0:
        fiveYearsAgo = current_time - timedelta(days = ctr*365+2)
        endDate = fiveYearsAgo + timedelta(days=5)
        try:
            df = stock.history(start=fiveYearsAgo.strftime("%Y-%m-%d"), end=endDate.strftime("%Y-%m-%d"), interval="1d")
            stock_price_5y_ago = df.iloc[0]['Close']

            if stock.info.get('currency') != 'USD':
                if stock.info.get('currency') == 'GBp':
                    currency = yfc.Ticker('GBPUSD=X')
                elif stock.info.get('currency') == 'CHF':
                    currency = yfc.Ticker('CHFUSD=X')
                elif stock.info.get('currency') == 'EUR':
                    currency = yfc.Ticker('EURUSD=X')
                elif stock.info.get('currency') == 'TWD':
                    currency = yfc.Ticker('TWDUSD=X')
                elif stock.info.get('currency') == 'DKK':
                    currency = yfc.Ticker('DKKUSD=X')
                elif stock.info.get('currency') == 'KRW':
                    currency = yfc.Ticker('KRWUSD=X')
                elif stock.info.get('currency') == 'HKD':
                    currency = yfc.Ticker('HKDUSD=X')
                elif stock.info.get('currency') == 'JPY':
                    currency = yfc.Ticker('JPYUSD=X')
                elif stock.info.get('currency') == 'CAD':
                    currency = yfc.Ticker('CADUSD=X')
                elif stock.info.get('currency') == 'AUD':
                    currency = yfc.Ticker('AUDUSD=X')
                elif stock.info.get('currency') == 'SEK':
                    currency = yfc.Ticker('SEKUSD=X')
                elif stock.info.get('currency') == 'SGD':
                    currency = yfc.Ticker('SGDUSD=X')
                elif stock.info.get('currency') == 'ILA':
                    currency = yfc.Ticker('ILSUSD=X')
                elif stock.info.get('currency') == 'NOK':
                    currency = yfc.Ticker('NOKUSD=X')
                elif stock.info.get('currency') == 'NZD':
                    currency = yfc.Ticker('NZDUSD=X')
                elif stock.info.get('currency') == 'INR':
                    currency = yfc.Ticker('INRUSD=X')
                elif stock.info.get('currency') == 'SAR':
                    currency = yfc.Ticker('SARUSD=X')
                elif stock.info.get('currency') == 'BRL':
                    currency = yfc.Ticker('SARUSD=X')
                elif stock.info.get('currency') == 'ZAc':
                    currency = yfc.Ticker('ZARUSD=X')
                elif stock.info.get('currency') == 'IDR':
                    currency = yfc.Ticker('IDRUSD=X')
                elif stock.info.get('currency') == 'IDR':
                    currency = yfc.Ticker('IDRUSD=X')
                elif stock.info.get('currency') == 'KWF':
                    currency = yfc.Ticker('KWDUSD=X')
                else:
                    print("Unknown currency: " + stock.info.get('currency'))

                df_currency = currency.history(start=fiveYearsAgo.strftime("%Y-%m-%d"), end=endDate.strftime("%Y-%m-%d"), interval="1d")
                initialRate = df_currency.iloc[0]['Close']
                endingRate = currency.info['previousClose']
            ctr = 0
        except Exception:
            ctr -=1

    dividends = stock.dat.dividends

    return (
        # using avg of rate for dividends and 40% tax, 0% tax for capital gain
        (dividends.loc[fiveYearsAgo.strftime("%Y-%m-%d"):current_time.strftime("%Y-%m-%d")].sum()*(endingRate+initialRate)/2)*0.6 + 
        (endingRate*stock.info['previousClose']) -
        (initialRate*stock_price_5y_ago)
            )/(stock_price_5y_ago*initialRate)

# Function to calculate dividend growth rate
def calculate_dividend_growth_rate(stock):
    dividends = stock.dat.dividends

    resampled = dividends.resample('YE').sum()
    if len(resampled) < 1:
        return 0
    
    if resampled.index[-1].date() > date.today():
        # bypassing the current year as some additional dividends might come and it would break the growth rate 
        idx = -2
    else:
        idx = -1

    if len(resampled) < abs(idx) + 5:
        return 0
    
    return ((resampled.iloc[idx] / resampled.iloc[idx-5]) ** (1 / 5) - 1) * 100

# Function to filter bad dividends data in Yahoo Finance
def filter_wrong_dividends(df):
  prevValue = -1
  prevIndex = -1

  filteredDf = pd.DataFrame(columns = df.columns,
                   index = df.index)

  for index, row in df.iterrows():
    if (prevValue == -1) :
          prevValue = row['Dividends']
          prevIndex = index
          filteredDf.loc[index] = row
    else:
        #filtering dividend data if they are duplicated in less then 2 days
        if (prevValue == row['Dividends']) and (index - prevIndex < timedelta(days=2)):
            print(index)
            print(prevIndex)
            print(prevValue)
            print(row['Dividends'])
        else:
          filteredDf.loc[index] = row

        prevIndex = index
        prevValue = row['Dividends']

  filteredDf = filteredDf.dropna()

  return filteredDf

# Function to calculate dividend streak
def calculate_dividend_streak(stock):
    dividends_df = stock.dat.dividends.to_frame()
    filtered_dividends_df = filter_wrong_dividends(dividends_df)
    dividends = filtered_dividends_df.squeeze()

    if type(dividends) != pd.Series :
        return 0

    resampled = dividends.resample('YE').sum()

    #return 0 if there is no dividend data or if the last dividend happened more than 1 year ago
    if (len(resampled) < 1) or (date.today() - resampled.index[-1].date() > pd.Timedelta(days=365)):
        return 0
    
    prevprev = 0
    prev = 0
    ctr = 0
    for i in range(len(resampled)-1):
        if (resampled.iloc[i] >= prev) and (prev > 0):
            if (i == 0) or (resampled.index[i].date()- resampled.index[i-1].date() < pd.Timedelta(days=380)):
                ctr += 1
        else:
            # try to support the case where there is exceptional dividends in the middle of a streak
            if(i <= len(resampled)-1):
                if ((resampled.iloc[i+1] > prev) and (prev > 0)) or ((resampled.iloc[i] > prevprev) and (prevprev > 0)):
                    if (i == 0) or (resampled.index[i].date()- resampled.index[i-1].date() < pd.Timedelta(days=380)):
                        ctr += 1
                    else:
                        ctr = 0
                else:
                    ctr = 0
            else:
                ctr = 0
        #print(ctr, resampled.index[i].date())
        prevprev = prev
        prev = resampled.iloc[i]

    return ctr

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

# Read stock tickers from file
def read_stock_tickers(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


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
        row['5y_total_return'] > spy_return,
        row['earning_growth'] > 0,
        row['w13612'] > 0
    ])
    return positives

# Highlight positive and negative metrics
def highlight_metrics(row, df):
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
            stock = yfc.Ticker('SPY')
            spy_return = round(calculate_5y_total_return_rate(stock), 2)
            colors.append('background-color: green' if row[col] > spy_return else 'background-color: red')
        elif col == 'earning_growth':
            colors.append('background-color: green' if row[col] > 0 else 'background-color: red')
        else:
            colors.append('')
    return colors

    

@st.cache_data(ttl=3600*24)
def load_data(ticker_file):
    stock_tickers = read_stock_tickers(ticker_file)
    stock_tickers_amount = len(stock_tickers)

    ctr = 0
    stock_data = []
    for ticker in stock_tickers:
        stock_data.append(fetch_stock_data(ticker, ctr/stock_tickers_amount))
        ctr+=1

    stock_data = [data for data in stock_data if data is not None]  # Filter out stocks with missing data
    stock_df = pd.DataFrame(stock_data)

    stock = yfc.Ticker('SPY')
    spy_return = round(calculate_5y_total_return_rate(stock), 2)

    stock_df['Score'] = stock_df.apply(lambda row: calculate_metrics(row, spy_return), axis=1, result_type="expand")

    sorted_df = stock_df.dropna()  # Remove rows with any missing data
    sorted_df = sorted_df.round(2)  # Round to two decimal places

    return sorted_df

def main():
    st.set_page_config(page_title="Stock Screener", layout="wide")
    st.title("Stock Screener")
    global progress_bar
    progress_bar = st.progress(0, text="Operation in progress. Please wait.")

    sorted_df = load_data(sys.argv[1])

    progress_bar.empty()
    
    df = filter_dataframe(sorted_df)
    st.dataframe(df.style.apply(highlight_metrics, axis=1, args=(df,)), use_container_width = True)


if __name__ == "__main__":
    main()