# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:37:23 2019

@author: Richard Hardis
"""
import numpy as np
import pandas as pd
import smtplib
import time

from datetime import datetime as dt
from datetime import timedelta as td

from alpha_vantage.timeseries import TimeSeries

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class securityData:
    def __init__(self):
        #self.tickers = ['spy','khc','ba','mmm','xlu','xlk','xlf','xlb','iwm','qqq','xlv','bsbr','enia','hpq','kdp']
        self.tickers = self.get_tickers_from_csv(['C:\\Users\\Richard Hardis\\Documents\\GitHub\\FinPy\\march-strategy\\symbol_list_NYSE.csv'])
        self.macd_window = [5,15,3]
        self.stoch_window = [200,3,7]
        self.buy_criteria = 5
        self.sell_criteria = 95
        self.distribution_list = ['richardphardis@gmail.com', 'samkest419@gmail.com']
        self.K_window = 3
        self.D_window = 7

    
    def get_tickers_from_csv(self, csv_list):
        for csv in csv_list:
            df = pd.read_csv(csv)
            
        symbol_list = []
        symbol_list = df.Symbol
        #symbol_list = symbol_list[:1]
            
        return symbol_list
    

def main(start_date=None, end_date=None, send_email=False):
    # 1. Create an instance of the securityData class
    sd = securityData()
    
    # 2. Loop over all of the tickers in sd and calculate the stochastic value for each
    tlist = []
    tvalues = []
    for ticker in sd.tickers:
        print(ticker)
        tlist.append(ticker)
        
        ticker_data = get_ticker_data(ticker, 'daily', '0')
        if start_date and end_date:
            try:
                prior_data = constrain_data(ticker_data, None, start_date)
                post_data = constrain_data(ticker_data, start_date, end_date)
                tvalues.append(calculate_stochastic(prior_data, sd.macd_window, sd.stoch_window))
            except:
                print('Invalid dates.  Try a later start date')
        else:
            try:
                tvalues.append(calculate_stochastic(ticker_data, sd.macd_window, sd.stoch_window))
            except:
                print('there was an error in calculating the stochastic')
        
        
    
    # 3. Combine all results into a single dataframe
    combined_series = pd.Series(tvalues, tlist)
    
    
    # 4. Filter the Series based on our desired cutoff values.
    combined_series_buy = combined_series[combined_series < sd.buy_criteria]
    combined_series_sell = combined_series[combined_series > sd.sell_criteria]
    
    print('buy:')
    for item in combined_series_buy:
        print(item)
        
    print('sell:')
    for item in combined_series_sell:
        print(item)
        
        
    
    # 5. Send an email containing the list of all of the securities that match the criteria
    distribution_list = ['richardphardis@gmail.com','samkest419@gmail.com']
    subject = 'Buy signals for the day of {}'.format(dt.now())
    
    message = ''
    for item in combined_series_buy.index:
        message = message + 'Buy ticker {}\n'.format(item)
        
    for item in combined_series_sell.index:
        message = message + 'Sell ticker {}\n'.format(item)
        
    message = message + '\n\n$$$'
        
    print(message)
    
    if send_email:
        for email in distribution_list:
            print('emailing to {}'.format(email)) 
            email_blast(email, subject, message)

      
def constrain_data(df, start_date, end_date):
    if end_date:
        df = df[df.DT <= end_date]
    
    if start_date:
        df = df[df.DT >= start_date]

    return df


def get_ticker_data(ticker, pull_type, interval='0', first_flag=False, ts=None):
    df_flag = False
    count = 0
    while not df_flag and count < 10:
        try:
            ticker_df = pull_data(ticker, pull_type, interval=interval)
            ticker_df = ticker_df.iloc[:,:]
            df_flag = True
        except KeyError:
            print('Pulled Too Soon!  Wait 2 seconds')
            time.sleep(2)
            
        count += 1
        
    if not df_flag:
        ticker_df = pd.DataFrame()
    
    if first_flag:    
        ticker_df = convert_str_to_dt(ticker_df)
    else:
        ticker_df.DT = ts
    
    t_series = ticker_df.DT
    
    return ticker_df, t_series


def convert_str_to_dt(df):
    for row in range(len(df.index)):
        df.DT[row] = dt.strptime(df.DT[row], '%Y-%m-%d')
    
    return df


def calculate_stochastic(ticker_df, macd_args, stoch_args):
    """
    Calculates the stochastic value of the provided ticker and returns a list with ticker name and value
    
    Args:
        ticker_df (pd.DataFrame): dataframe containing price and volume data about a ticker symbol
        macd_args (list): a list of the three macd values
        stoch_args (list): a list of the three stochastic values
        
    Returns:
        stoch_val (float): the numeric value of the stochastic
    """
    
    # Get the MACD dataframe
    df_macd = create_macd(ticker_df, *macd_args)
    
    # Run some math on the ticker dataframe
    lookback_val = len(df_macd) if len(df_macd) < stoch_args[0] else stoch_args[0]
    df_macd = df_macd.iloc[-lookback_val:,:]
    max_val = df_macd.signal.max()
    min_val = df_macd.signal.min()
    
    df_macd['K_200'] = (df_macd.signal - min_val) / (max_val - min_val) * 100
    K_Full = np.mean(df_macd.K_200[-stoch_args[1]:])
    #print(K_Full)
    
    D_Avg = np.mean(df_macd.K_200[-stoch_args[2]:])
    #print(D_Avg)
    
    stoch_val = K_Full
    
    return stoch_val


def email_blast(to_address, subject, message):
    conn = smtplib.SMTP('smtp.gmail.com',587)
    conn.ehlo()
    conn.starttls()
    conn.login('Rich.Sam.Signals@gmail.com','Login435!rss')
    conn.sendmail('Rich.Sam.Signals@gmail.com',to_address,'Subject: {}\n\n{}'.format(subject,message))
    print('emails away!')
    

def create_macd(df,span1,span2,span3):
    df = df.copy(deep=True)
    df['stock_fast_ema'] = pd.ewma(df['Close'], span=span1)
    df['stock_slow_ema'] = pd.ewma(df['Close'], span=span2)
    df['macd'] = df['stock_fast_ema'] - df['stock_slow_ema']
    df['signal'] = pd.ewma(df['macd'], span=span3)
    df['crossover'] = df['macd'] - df['signal'] # means, if this is > 0, or stock_df['Crossover'] =  stock_df['MACD'] - stock_df['Signal'] > 0, there is a buy signal                                                                     # means, if this is < 0, or stock_df['Crossover'] =  stock_df['MACD'] - stock_df['Signal'] < 0, there is a sell signal
    return df


def pull_data(ticker,pullType,interval='0',key='1RJDU8R6RESLVE09'):
    ts = TimeSeries(key=key, output_format='pandas')
    
    if pullType == 'intraday':
        data1, meta_data1 = ts.get_intraday(symbol=ticker,interval=interval,outputsize='full')
        data1.columns = ['Open','High','Low','Close','Volume']
        
        if interval == '1min':
            td = 1
        elif interval == '5min':
            td = 5
        elif interval == '15min':
            td = 15
        elif interval == '30min':
            td = 30
        elif interval == '60min':
            td = 60
        else:
            td = 0
        
        data1['TimeDelta'] = td*60*1000*.75
        
        print('Pulled intraday with interval = ' + interval + '\n')
    elif pullType == 'daily':
        data1, meta_data1 = ts.get_daily_adjusted(symbol=ticker,outputsize='full')
        data1=data1.iloc[:,[0,1,2,3,5,6,7]]
        data1.columns = ['Open','High','Low','Close','Volume','Dividend','Split Coef']
        data1['TimeDelta'] = 24*60*60*1000*.75
        print('Pulled daily\n')
    elif pullType == 'weekly':
        data1, meta_data1 = ts.get_weekly_adjusted(symbol=ticker)
        data1=data1.iloc[:,[0,1,2,3,5,6]]
        data1.columns = ['Open','High','Low','Close','Volume','Dividend']
        data1['TimeDelta'] = 7*24*60*60*1000*.75
        print('Pulled weekly\n')
    elif pullType == 'monthly':
        data1, meta_data1 = ts.get_monthly_adjusted(symbol=ticker)
        data1=data1.iloc[:,[0,1,2,3,5,6]]
        data1.columns = ['Open','High','Low','Close','Volume','Dividend']
        data1['TimeDelta'] = 30*24*60*60*1000*.75
        print('Pulled monthly\n')
    else:
        print('Please enter a valid pull type')

    data1['DT'] = data1.index
    
    return data1


if __name__ == '__main__':
    main(send_email=False)
