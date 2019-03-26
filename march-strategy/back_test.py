# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:02:27 2019

@author: Richard Hardis
"""
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import run_strat as rs


def downcast_floats(df):
    df_copy = df.copy(deep=True)
    df_float = df_copy.select_dtypes(include=['float'])
    converted_float = df_float.apply(pd.to_numeric, downcast='float')
    df[converted_float.columns] = converted_float
    
    return df


def calculate_return(df, buy_date, period):
    df = df.copy(deep=True)
    df.reset_index(inplace=True)
    df = rs.constrain_data(df, buy_date, end_date)
    start_df = df[df.DT == buy_date]
    start_list = start_df['Close']
    start_price = start_list.iloc[0]
    start_date_index = df[df.DT == buy_date].index.values.astype(int)[0]
    end_price = df['Close'][start_date_index + period]
    sell_date = df.DT[start_date_index + period]
    if (sell_date - df.DT.iloc[-1]).days > 0:
        print('end date out of range')
        net_return = np.nan
    else:
        net_return = (end_price - start_price) / start_price * 100
    
    return [start_price, end_price, sell_date, net_return]

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
for i, ticker in enumerate(['SPY']):#sd.tickers[:500]: # this is normally ticker in sd.tickers to get all tickers on NYSE
    try:
        print(ticker)
        if i == 0:
            unconstrained_data, ts = rs.get_ticker_data(ticker.upper(), 'daily', '0', True, None)
            print('changed strings to datetime')
        else:
            unconstrained_data, unused = rs.get_ticker_data(ticker.upper(), 'daily', '0', False, ts)
            print('replaced strings with datetime from ticker 0')
            
        unconstrained_data = downcast_floats(unconstrained_data)
        
        dates_df = unconstrained_data[(unconstrained_data.DT >= start_date) & (unconstrained_data.DT <= end_date)]
        for row in dates_df.itertuples():
            day = row.DT
            ticker_data = rs.constrain_data(unconstrained_data, None, day)
            stoch_val = rs.calculate_stochastic(ticker_data, sd.macd_window, sd.stoch_window)
            if stoch_val < 5:
                df_list = [ticker]
                df_list.append(day)
                df_list.append(stoch_val)
                df_list.extend(calculate_return(unconstrained_data, day, days_until_sale))
                greater_df_list.append(df_list)
            elif stoch_val > 95:
                df_list = [ticker]
                df_list.append(day)
                df_list.append(stoch_val)
                #df_list.append('Sell_Signal')
                df_list.extend(calculate_return(unconstrained_data, day, days_until_sale))
                #df_list.append(np.nan)
                greater_df_list.append(df_list)       
    except:
        print('could not load data for {}'.format(ticker))
        
            
returns_df = pd.DataFrame(greater_df_list, columns=['ticker','buy_date','stochastic_value', 'buy_price','sell_price','sell_date', 'return'])
returns_df = returns_df[returns_df.stochastic_value <= 5]
returns_df.to_csv('returns_summary.csv')
            
end_time = datetime.now()

elapsed_time = end_time - start_time
print(elapsed_time)
        
        
    
    