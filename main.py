import yfinance as yf
import pandas as pd
import streamlit as st
import sys
from datetime import date, datetime, timedelta

# Function to fetch stock data
def fetch_stock_data(ticker):
    print(f'Fetching data for {ticker}')
    stock = yf.Ticker(ticker)
    info = stock.info

    fiveYearAvgDividendYield = stock.info.get('fiveYearAvgDividendYield')
    dividendRate = stock.info.get('dividendRate')
    close = stock.info.get('previousClose')
    fiftyDayAverage = info.get('fiftyDayAverage', '')
    twoHundredDayAverage = info.get('twoHundredDayAverage', '')
    if close is None or close == 'N/A':
        close = 1
    if fiftyDayAverage is None or fiftyDayAverage == '':
        fiftyDayAverage = 1
    if twoHundredDayAverage is None or twoHundredDayAverage == '':
        twoHundredDayAverage = 1    
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
            '50d-SMA': round(((close - fiftyDayAverage)/close), 2), 
            '200d-SMA': round(((close - twoHundredDayAverage)/close), 2),
        }
    except KeyError:
        return None

# Function to calculate 5y total return rate
def calculate_5y_total_return_rate(stock):
    current_time = datetime.now()
    ctr = 5
    initialRate = 1
    endingRate = 1
    while True:
        fiveYearsAgo = current_time - timedelta(days = ctr*365+2)
        endDate = fiveYearsAgo + timedelta(days=5)
        df = stock.history(start=fiveYearsAgo.strftime("%Y-%m-%d"), end=endDate.strftime("%Y-%m-%d"), interval="1d")

        if stock.info.get('currency') != 'USD':
            if stock.info.get('currency') == 'GBp':
                currency = yf.Ticker('GBPUSD=X')
            elif stock.info.get('currency') == 'CHF':
                currency = yf.Ticker('CHFUSD=X')
            elif stock.info.get('currency') == 'EUR':
                currency = yf.Ticker('EURUSD=X')
            elif stock.info.get('currency') == 'TWD':
                currency = yf.Ticker('TWDUSD=X')
            elif stock.info.get('currency') == 'DKK':
                currency = yf.Ticker('DKKUSD=X')
            elif stock.info.get('currency') == 'KRW':
                currency = yf.Ticker('KRWUSD=X')
            elif stock.info.get('currency') == 'HKD':
                currency = yf.Ticker('HKDUSD=X')
            elif stock.info.get('currency') == 'JPY':
                currency = yf.Ticker('JPYUSD=X')
            elif stock.info.get('currency') == 'CAD':
                currency = yf.Ticker('CADUSD=X')
            elif stock.info.get('currency') == 'AUD':
                currency = yf.Ticker('AUDUSD=X')
            elif stock.info.get('currency') == 'SEK':
                currency = yf.Ticker('SEKUSD=X')
            elif stock.info.get('currency') == 'SGD':
                currency = yf.Ticker('SGDUSD=X')
            elif stock.info.get('currency') == 'ILA':
                currency = yf.Ticker('ILSUSD=X')
            elif stock.info.get('currency') == 'NOK':
                currency = yf.Ticker('NOKUSD=X')
            elif stock.info.get('currency') == 'NZD':
                currency = yf.Ticker('NZDUSD=X')
            else:
                print("Unknown currency: " + stock.info.get('currency'))

            df_currency = currency.history(start=fiveYearsAgo.strftime("%Y-%m-%d"), end=endDate.strftime("%Y-%m-%d"), interval="1d")
            initialRate = df_currency.iloc[0]['Close']
            endingRate = currency.info['previousClose']

        if len(df)==0:
            ctr -=1
        else:
            break

    dividends = stock.dividends

    return (
        (dividends.loc[fiveYearsAgo.strftime("%Y-%m-%d"):current_time.strftime("%Y-%m-%d")].sum()*(endingRate+initialRate)/2) + 
        (endingRate*stock.info['previousClose']) -
        (initialRate*df.iloc[0]['Close'])
            )/(df.iloc[0]['Close']*initialRate)

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
    stock = yf.Ticker('SPY')
    spy_return = round(calculate_5y_total_return_rate(stock), 2)

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
        row['earning_growth'] > 0
    ])
    negatives = 11 - positives
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
        elif col == '5y_total_return':
            stock = yf.Ticker('SPY')
            spy_return = round(calculate_5y_total_return_rate(stock), 2)
            colors.append('background-color: green' if row[col] > spy_return else 'background-color: red')
        elif col == 'earning_growth':
            colors.append('background-color: green' if row[col] > 0 else 'background-color: red')
        else:
            colors.append('')
    return colors

# Remove the index column
st.dataframe(sorted_df.style.apply(highlight_metrics, axis=1))