# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:02:27 2019

@author: Richard Hardis
"""
import numpy as np
import pandas as pd
import time

from datetime import datetime, timedelta

import run_strat as rs


def downcast_floats(df):
    df_copy = df.copy(deep=True)
    df_float = df_copy.select_dtypes(include=['float'])
    converted_float = df_float.apply(pd.to_numeric, downcast='float')
    df[converted_float.columns] = converted_float
    
    return df


def calculate_return(df, period):
    df = df.copy(deep=True)
    df.reset_index(inplace=True)
    
    df['day_offset'] = df.Close.shift(-period)
    df['returns'] = (df.day_offset - df.Close) / df.Close * 100
    
    returns_df = df.drop(['day_offset'], axis=1)
    #returns_df = df
    
    return returns_df

start_time = datetime.now()

start_date = datetime(1999, 1, 1)  # Change this line to change the start date you want to use. (Year, Month, Day)
end_date = datetime.now()
days_until_sale = 10
#delta_dates = end_date - start_date
#dates_list = []
#for i in range(delta_dates.days):
#    dates_list.append(start_date + timedelta(days=i))

sd = rs.securityData()
greater_df_list = []
temp_list = ['SPY','MCK','AAPL','GOOG','KHC','UTX','IWN','XLB','XLE','JNJ','BAC']
summary_df = pd.DataFrame()
for i, ticker in enumerate(sd.tickers[:3]):#sd.tickers[:500]: # this is normally ticker in sd.tickers to get all tickers on NYSE
    print(ticker)
    time.sleep(2)
    try:
        if i == 0:
            unconstrained_data, ts = rs.get_ticker_data(ticker.upper(), 'daily', '0', True, None)
            print('changed strings to datetime')
            summary_df = pd.DataFrame(columns=unconstrained_data.columns)
            summary_df['ticker'] = ticker
        else:
            unconstrained_data, _ = rs.get_ticker_data(ticker.upper(), 'daily', '0', False, ts)
            print('replaced strings with datetime from ticker 0')
            
        unconstrained_data = downcast_floats(unconstrained_data)
        
        # run all strategies
        stoch_df = rs.calculate_stochastic(unconstrained_data, sd.macd_window, sd.stoch_window)
        
        returns_df = calculate_return(stoch_df, days_until_sale)
        
        returns_df = returns_df[(returns_df.stoch_indicator < 5) & (returns_df.stoch_indicator > 0)]
        returns_df['ticker'] = ticker
        
        summary_df = pd.concat([summary_df, returns_df])
        
        returns_df.to_csv('returns_summary_{}.csv'.format(ticker))
    except:
        print('Something went wrong with ticker {}'.format(ticker))


            
end_time = datetime.now()

elapsed_time = end_time - start_time
print(elapsed_time)
        
        
    
    