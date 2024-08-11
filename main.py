import yfinance as yf
import pandas as pd
import streamlit as st
import sys
from datetime import date,datetime, timedelta

# Function to fetch stock data
def fetch_stock_data(ticker):
    print(f'Fetching data for {ticker}')
    stock = yf.Ticker(ticker)
    info = stock.info

    fiveYearAvgDividendYield = stock.info.get('fiveYearAvgDividendYield')
    dividendRate = stock.info.get('dividendRate')
    close = stock.info.get('previousClose')
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
            'dividend_streak': calculate_dividend_streak(stock),
            'dividend_yield': round(info.get('dividendYield', 0) * 100, 2),
            '5y_Agv_dividend_yield': round(fiveYearAvgDividendYield, 2),
            'payout_ratio': round(info.get('payoutRatio', 0), 2),
            'dividend_growth_rate': round(calculate_dividend_growth_rate(stock), 2),
            'eps': round(info.get('trailingEps', 0), 2),
            'pe_ratio': round(info.get('trailingPE', 0), 2),
            'debt_to_equity': round(info.get('debtToEquity', 0), 2),
            'roe': round(info.get('returnOnEquity', 0), 2),
            'discount/premium':round(discount,2)
        }
    except KeyError:
        return None

# Function to calculate dividend growth rate
def calculate_dividend_growth_rate(stock):
    dividends = stock.dividends

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
    dividends_df = stock.dividends.to_frame()
    filtered_dividends_df = filter_wrong_dividends(dividends_df)
    dividends = dividends_df.squeeze()

    if type(dividends) != pd.Series :
        return 0

    resampled = dividends.resample('YE').sum()
    if len(resampled) < 1:
        return 0
    
    prev = 0
    prevprev = 0
    ctr = 0
    for i in range(len(resampled)-1):
        if resampled.iloc[i] >= prev:
            ctr += 1
        else:
            # try to support the case where there is exceptional dividends in the middle of a streak
            if resampled.iloc[i] >= prevprev and prevprev > 0:
                ctr += 1
            else:
                ctr = 0
        prevprev = prev
        prev = resampled.iloc[i]

    return ctr

# Read stock tickers from file
def read_stock_tickers(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

# Fetch data for stocks
stock_tickers = read_stock_tickers(sys.argv[1])
stock_data = [fetch_stock_data(ticker) for ticker in stock_tickers]
stock_data = [data for data in stock_data if data is not None]  # Filter out stocks with missing data
stock_df = pd.DataFrame(stock_data)

# Calculate positive and negative metrics
def calculate_metrics(row):
    positives = sum([
        row['dividend_yield'] > 3,
        0.30 <= row['payout_ratio'] <= 0.60,
        row['dividend_growth_rate'] > 5,
        row['eps'] > 0,
        row['pe_ratio'] < 20,
        row['debt_to_equity'] < 100,
        row['roe'] > 0.10,
        row['discount/premium'] < 0
    ])
    negatives = 8 - positives
    return positives, negatives

stock_df[['positives', 'negatives']] = stock_df.apply(lambda row: calculate_metrics(row), axis=1, result_type="expand")

# Streamlit interface
st.title("Dividend Stock Screener")

st.write("Select sorting criteria:")
criteria = st.selectbox("Sort by", ["Dividend Yield", "Payout Ratio", "Dividend Growth Rate", "EPS", "P/E Ratio", "Debt-to-Equity Ratio", "ROE", "positives"])

sorted_df = stock_df.sort_values(by=criteria.replace(" ", "_").lower(), ascending=False)
sorted_df = sorted_df.dropna()  # Remove rows with any missing data
sorted_df = sorted_df.round(2)  # Round to two decimal places

# Highlight positive and negative metrics
def highlight_metrics(row):
    colors = []
    for col in sorted_df.columns:
        if col == 'dividend_yield':
            colors.append('background-color: green' if row[col] > 3 else 'background-color: red')
        elif col == 'payout_ratio':
            colors.append('background-color: green' if 0.30 <= row[col] <= 0.60 else 'background-color: red')
        elif col == 'dividend_growth_rate':
            colors.append('background-color: green' if row[col] > 5 else 'background-color: red')
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
        else:
            colors.append('')
    return colors

# Remove the index column
st.dataframe(sorted_df.style.apply(highlight_metrics, axis=1))