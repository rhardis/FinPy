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


def retrieve(i, ticker, from_csv):
    print(ticker)
    if from_csv:
        unconstrained_data = pd.read_csv('{}_unc.csv'.format(ticker))
        unconstrained_data['DT'] = pd.to_datetime(unconstrained_data.DT)
    else:
        #try:
        if i == 0:
            unconstrained_data, ts = rs.get_ticker_data(ticker.upper(), timeframe, '0', True, None)
            print('changed strings to datetime')
            summary_df = pd.DataFrame(columns=unconstrained_data.columns)
            summary_df['ticker'] = ticker
        else:
            unconstrained_data, _ = rs.get_ticker_data(ticker.upper(), timeframe, '0', False, ts)
            print('replaced strings with datetime from ticker 0')
            
        #unconstrained_data = unconstrained_data.iloc[-100:,:]    
    return unconstrained_data


def calculate_return(df, period):
    df = df.copy(deep=True)
    df.reset_index(inplace=True)
    
    df['day_offset'] = df.Close.shift(-period)
    df['returns'] = (df.day_offset - df.Close) / df.Close * 100
    
    returns_df = df.drop(['day_offset'], axis=1)
    #returns_df = df
    
    return returns_df

def calculate_exp_ret(ev_df, trades_df, num_top):
    ev_df_new_index = ev_df.copy(deep=True)
    ev_df_new_index = ev_df_new_index.set_index(ev_df_new_index.Ticker)
    trades_df['Expected_Value_Top{}'.format(num_top)] = np.nan
    trades_df['trade_return_multiple'] = np.nan
    
    for i, row in trades_df.iterrows():
        if row.trade_available > 0:
            tlist = row.index[:-4]
            buy_list = [idx for idx in tlist if ((row.loc[idx] == True))]
            buy_list = [idx[:-4] for idx in buy_list if ('buy' in idx)]
            ev_list = []
            for ticker in buy_list:
                append_val = ev_df_new_index.loc[ticker, 'Expected_Return']
                ev_list.append(append_val)
            
            if len(ev_list) > 1:
                ev_list = sorted(ev_list, reverse=True)
            if len(ev_list) > num_top:
                ev_list = ev_list[:num_top]
            trades_df.loc[i, 'Expected_Value_Top{}'.format(num_top)] = np.mean(ev_list)
            
            #print([tick for tick in buy_list])
            mean_cols_list = ['{}_return_multiplier'.format(tick) for tick in buy_list]
            #print(mean_cols_list)
            trades_df.loc[i, 'trade_return_multiple'] = np.mean(row[mean_cols_list])
        else:
            continue
    
    return trades_df


start_time = datetime.now()

start_date = datetime(1999, 1, 1)  # Change this line to change the start date you want to use. (Year, Month, Day)
end_date = datetime.now()
days_until_sale = 10


sd = rs.securityData()
greater_df_list = []
temp_list = ['SPY','MCK','AAPL','GOOG','KHC','UTX','IWN','XLB','XLE','JNJ','BAC']
summary_df = pd.DataFrame()

exit_percent = 1
exit_low_percent = -1
num_top = 3

test_strat_df = pd.DataFrame.from_dict({'Ticker':'remove',
                                        'Avg_Return_Sell_Low':[0],
                                        'Avg_Return_Sell_High':[0],
                                        'Stdv_Return_Sell_High':[0],
                                        'Avg_Return_Sell_Close':[0],
                                        'Stdv_Return_Sell_Close':[0],
                                        'Accuracy_{}%'.format(exit_percent):[0],
                                        'Expected_Return':[0],
                                        'Buy_Frequency':[0]})
timeframe = 'daily'

ticker_list = sd.tickers[:50]
tlist_size = len(ticker_list)
t3_count = int(np.floor(tlist_size/3))
count_list = np.arange(t3_count+1)


i = 0
for counter in count_list:
    #time.sleep(30)
    if counter == t3_count+1:
        tlist = ticker_list[-3:]
    else:
        tlist = ticker_list[3 * counter: 3 * counter + 3]
    
    #print(tlist)
    #print('\n')
    
    for ticker in tlist:#sd.tickers[:500]: # this is normally ticker in sd.tickers to get all tickers on NYSE
        if ticker in ['BF.B', 'NFX', 'PX', 'WYN', 'XY']:
            continue
        
        unconstrained_data = retrieve(i, ticker, from_csv=True)
        #unconstrained_data = unconstrained_data.iloc[-500:,:]
        if i == 0:
            all_df = unconstrained_data.copy(deep=True)
            i = 1
        #print(type(unconstrained_data.DT[0]))
        
        if len(unconstrained_data) == 0:
            #print('could not get data for {}'.format(ticker))
            continue
        #unconstrained_data.to_csv('{}_unc.csv'.format(ticker))
        
        
        # run all strategies
        #all_df = rs.calculate_stochastic(unconstrained_data, sd.macd_window, sd.stoch_window)
        #all_df = rs.calc_keltner(all_df, 5, .6, 3, 1)
        wdf = rs.ichi_open_cross(unconstrained_data, 5, exit_percent, exit_low_percent)
        
        #wdf.to_csv('{}_results.csv'.format(ticker))
        
        ldf = wdf[(wdf.buy_bool == True) & (wdf.percent_sale_bool == False)]
        #print('ldf = {}'.format(len(ldf)))
        #print('wdf potential buys = {}'.format(len(wdf[wdf.buy_bool])))
        avg_best_return = np.mean(wdf.return_over_projected)
        av_std_best_return = np.std(wdf.return_over_projected)
        avg_close_return = np.mean(wdf.return_sold_close)
        av_std_close_return = np.std(wdf.return_sold_close)
        
        try:
            num_correct = len(wdf[(wdf.percent_sale_bool) & (wdf.buy_bool) & ~(wdf.exit_low_bool)])
            num_loss = len(wdf[(wdf.exit_low_bool) & (wdf.buy_bool) & ~(wdf.percent_sale_bool)])
            #print('num correct for {} = {}'.format(ticker, num_correct))
            total = len(wdf[wdf.buy_bool])
            #print('new total = {}'.format(total))
            #print('total total = {}\n\n'.format(len(wdf)))
            accsetp = num_correct / total * 100
            accsetplow = num_loss / total * 100
        except:
            accsetp = 0
            
        exp_ret = (accsetp/100 * exit_percent) + (accsetplow/100 * exit_low_percent) + ((100-accsetp-accsetplow)/100 * np.mean(wdf.return_sold_close))
        bf = total/len(wdf) * 100
            
        all_df = all_df.reset_index(drop=True)
        all_df['{}_buy'.format(ticker)] = wdf.buy_bool
        all_df['{}_return_multiplier'.format(ticker)] = wdf.return_as_multiple
        
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
                                            'Avg_Return_Sell_High':[avg_best_return],
                                            'Stdv_Return_Sell_High':[av_std_best_return],
                                            'Avg_Return_Sell_Close':[avg_close_return],
                                            'Stdv_Return_Sell_Close':[av_std_close_return],
                                            'Accuracy_{}%'.format(exit_percent):[accsetp],
                                            'Freq_Loss_{}%'.format(exit_low_percent):[accsetplow],
                                            'Expected_Return':[exp_ret],
                                            'Buy_Frequency':[bf]})
        test_strat_df = pd.concat([test_strat_df, add_df])
        
        #returns_df.to_csv('returns_summary_{}.csv'.format(ticker))
    #    except:
    #        print('Something went wrong with ticker {}'.format(ticker))
        
#summary_df = summary_df.drop(['DT', 'Dividend'])

all_df['dates'] = all_df.DT
keep_cols = [col for col in all_df.columns if ('_buy' in col) or ('_return' in col)]
keep_cols.append('dates')
all_df = all_df[keep_cols]
#all_df = all_df.fillna(value=False)
sum_cols_list = [col for col in all_df.columns if ('buy' in col)]
all_df['trade_available'] = all_df[sum_cols_list].sum(axis=1)


days_of_trading = len(all_df[all_df.trade_available > 0])
dot_3plus = len(all_df[all_df.trade_available >= 3])
print('there were {} potential trade days out of {} total days.  {} of those days had >= 3 trades recommended.'.format(days_of_trading, len(all_df), dot_3plus))
print('Trading day coverage = {}%'.format((days_of_trading/len(all_df)*100)))
test_strat_df = test_strat_df.iloc[1:,:]
test_strat_df.reset_index()
test_strat_df = test_strat_df.sort_values(by=['Expected_Return'], ascending=False)

all_df  = calculate_exp_ret(test_strat_df, all_df, num_top)
total_return_multiple = all_df.trade_return_multiple.product()
print('Total Return Multiple = {}'.format(total_return_multiple))
pos = len(all_df[all_df.trade_return_multiple > 1])
neg = len(all_df[all_df.trade_return_multiple < 1])
print('There were {} positive trades out of {} total.  Percent = {}%'.format(pos, days_of_trading, pos/days_of_trading*100))
print('There were {} negative trades out of {} total.  Percent = {}%'.format(neg, days_of_trading, neg/days_of_trading*100))
            
end_time = datetime.now()

elapsed_time = end_time - start_time
print(elapsed_time)
