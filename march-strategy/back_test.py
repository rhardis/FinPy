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
from decompose import get_best_vars_percentile, normalize_df


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


sd = rs.securityData()
greater_df_list = []
temp_list = ['SPY','MCK','AAPL','GOOG','KHC','UTX','IWN','XLB','XLE','JNJ','BAC']
summary_df = pd.DataFrame()
test_strat_df = pd.DataFrame.from_dict({'Ticker':'remove',
                                        'Avg_Return_Sell_Low':[0],
                                        'Avg_Return_Sell_High':[0],
                                        'Stdv_Return_Sell_High':[0],
                                        'Avg_Return_Sell_Close':[0],
                                        'Stdv_Return_Sell_Close':[0],
                                        'Accuracy_Prev_Close':[0],
                                        'Accuracy_1%':[0]})
timeframe = 'daily'
for i, ticker in enumerate(sd.tickers[25:45]):#sd.tickers[:500]: # this is normally ticker in sd.tickers to get all tickers on NYSE
    print(ticker)
    time.sleep(2)
    #try:
    if i == 0:
        unconstrained_data, ts = rs.get_ticker_data(ticker.upper(), timeframe, '0', True, None)
        print('changed strings to datetime')
        summary_df = pd.DataFrame(columns=unconstrained_data.columns)
        summary_df['ticker'] = ticker
        all_df = unconstrained_data.copy(deep=True)
    else:
        unconstrained_data, _ = rs.get_ticker_data(ticker.upper(), timeframe, '0', False, ts)
        print('replaced strings with datetime from ticker 0')
        
    unconstrained_data = downcast_floats(unconstrained_data)
    #unconstrained_data = unconstrained_data.iloc[-100:,:]
    
    
    if len(unconstrained_data) == 0:
        print('could not get data for {}'.format(ticker))
        continue
    
    # run all strategies
    #all_df = rs.calculate_stochastic(unconstrained_data, sd.macd_window, sd.stoch_window)
    #all_df = rs.calc_keltner(all_df, 5, .6, 3, 1)
    wdf = rs.ichi_open_cross(unconstrained_data, 5, 3)
    ldf = wdf[(wdf.rose_to_projected == False) & (wdf.buy_bool == True)]
    avg_loss = np.mean(ldf.Loss_on_Low)
    avg_best_return = np.mean(wdf.return_over_projected)
    av_std_best_return = np.std(wdf.return_over_projected)
    avg_close_return = np.mean(wdf.return_sold_close)
    av_std_close_return = np.std(wdf.return_sold_close)
    try:
        num_correct = len(wdf[(wdf.next_high > wdf.prev_close) & (wdf.prev_close > wdf.projected_val)])
        total = len(wdf[wdf.prev_close > wdf.projected_val])
        accpc = num_correct / total * 100
    except:
        accpc = 0
    
    try:
        num_correct = len(wdf[wdf.percent_sale_bool & (wdf.prev_close > wdf.projected_val)])
        print('num correct for {} = {}'.format(ticker, num_correct))
        total = len(wdf[(wdf.prev_close > wdf.projected_val)])
        print('new total = {}'.format(total))
        print('total total = {}'.format(len(wdf)))
        acc1p = num_correct / total * 100
    except:
        acc1p = 0
        
    all_df['{}_met_criteria'.format(ticker)] = wdf.rose_to_projected
    #wdf = all_df[['Open', 'High', 'Low', 'Close', 'projected_val', 'next_high', 'prev_close', 'return_over_projected']]
    
    #returns_df = calculate_return(all_df, days_until_sale)
#    X = returns_df.copy(deep=True)
#    X = X.drop('date', axis=1)
#    X = X.dropna()
#    X = normalize_df(X)
#    y = X.pop('returns')
#    #best_scores = get_best_vars_percentile(X, y, 50)
#    model, score, ceofs, intercept = rs.make_model(X, y)
#    tts_score = rs.test_model_tts(X, y)
    
#    returns_df = returns_df[(returns_df.stoch_indicator < 5) & (returns_df.stoch_indicator > 0)]
#    returns_df['ticker'] = ticker
    
#    summary_df = pd.concat([summary_df, returns_df])
    
    add_df = pd.DataFrame.from_dict({'Ticker':ticker,
                                        'Avg_Return_Sell_Low':[avg_loss],
                                        'Avg_Return_Sell_High':[avg_best_return],
                                        'Stdv_Return_Sell_High':[av_std_best_return],
                                        'Avg_Return_Sell_Close':[avg_close_return],
                                        'Stdv_Return_Sell_Close':[av_std_close_return],
                                        'Accuracy_Prev_Close':[accpc],
                                        'Accuracy_1%':[acc1p]})
    test_strat_df = pd.concat([test_strat_df, add_df])
    
    #returns_df.to_csv('returns_summary_{}.csv'.format(ticker))
#    except:
#        print('Something went wrong with ticker {}'.format(ticker))
        
#summary_df = summary_df.drop(['DT', 'Dividend'])
all_df = all_df.iloc[:,8:]
all_df = all_df.fillna(value=False)
all_df['trade_available'] = all_df.sum(axis=1)
days_of_trading = len(all_df[all_df.trade_available > 0])
print('there were {} potential trade days out of {} total days'.format(days_of_trading, len(all_df)))
print(len(all_df[all_df.trade_available == 0]))
test_strat_df = test_strat_df.iloc[1:,:]
test_strat_df.reset_index()
            
end_time = datetime.now()

elapsed_time = end_time - start_time
print(elapsed_time)
