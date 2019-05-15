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
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from alpha_vantage.timeseries import TimeSeries

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class securityData:
    def __init__(self):
        #self.tickers = ['spy','khc','ba','mmm','xlu','xlk','xlf','xlb','iwm','qqq','xlv','bsbr','enia','hpq','kdp']
        self.tickers = self.get_tickers_from_csv(['C:\\Users\\Richard Hardis\\Documents\\GitHub\\FinPy\\march-strategy\\sp500constituents.csv'])
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
    while not df_flag and count < 3:
        try:
            ticker_df = pull_data(ticker, pull_type, interval=interval)
            ticker_df = ticker_df.iloc[:,:]
            df_flag = True
        except KeyError:
            waiting = 3
            print('Pulled Too Soon!  Wait {} seconds'.format(waiting))
            time.sleep(waiting)
            
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
    df.DT = pd.to_datetime(df.DT, infer_datetime_format=True)
    
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
    df_macd['max_val'] = df_macd.signal.rolling(lookback_val).max()
    df_macd['min_val'] = df_macd.signal.rolling(lookback_val).min()
    
    df_macd['K_200'] = (df_macd.signal - df_macd.min_val) / (df_macd.max_val - df_macd.min_val) * 100
    df_macd['stoch_val'] = df_macd.K_200.rolling(stoch_args[1]).mean()
    df_macd['shift_stoch'] = df_macd.stoch_val.shift(1)
    df_macd['sdiff_sign'] = np.sign(df_macd.stoch_val - df_macd.shift_stoch)
    df_macd['stoch_indicator'] = df_macd.sdiff_sign * df_macd.shift_stoch
    
    #df_stoch = df_macd.drop(['stoch_val', 'shift_stoch', 'sdiff_sign', 'Dividend', 'Split Coef', 'DT', 'stock_fast_ema', 'stock_slow_ema', 'macd', 'signal', 'crossover', 'max_val', 'min_val', 'K_200'], axis=1)
    df_stoch = df_macd
    
    return df_stoch


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
        
        
        print('Pulled intraday with interval = ' + interval + '\n')
    elif pullType == 'daily':
        data1, meta_data1 = ts.get_daily_adjusted(symbol=ticker,outputsize='full')
        data1=data1.iloc[:,[0,1,2,3,5,6,7]]
        data1.columns = ['Open','High','Low','Close','Volume','Dividend','Split Coef']
        print('Pulled daily\n')
    elif pullType == 'weekly':
        data1, meta_data1 = ts.get_weekly_adjusted(symbol=ticker)
        data1=data1.iloc[:,[0,1,2,3,5,6]]
        data1.columns = ['Open','High','Low','Close','Volume','Dividend']
        print('Pulled weekly\n')
    elif pullType == 'monthly':
        data1, meta_data1 = ts.get_monthly_adjusted(symbol=ticker)
        data1=data1.iloc[:,[0,1,2,3,5,6]]
        data1.columns = ['Open','High','Low','Close','Volume','Dividend']
        print('Pulled monthly\n')
    else:
        print('Please enter a valid pull type')

    data1['DT'] = data1.index
    
    return data1


def make_model(predictor_df, target_df):
    X = predictor_df
    y = target_df
    lm = linear_model.LinearRegression()
    model = lm.fit(X,y)
    
    return model, lm.score(X, y), lm.coef_, lm.intercept_


def test_model_tts(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    sc = model.score(X_test, y_test)
    
    return sc


def calc_keltner(df, ema_window, std_coef, atr_window, predicted_periods):
    df['HL'] = df.High - df.Low
    df['HCl'] = np.abs(df.High-df.Close.shift(1))
    df['LCl'] = np.abs(df.Low-df.Close.shift(1))
    df['TR'] = df.HL
    for idx, row in df.iterrows():
        try:
            max_ = np.max([row['HL'], row['HCl'], row['HCl']])
            print(max_)
            row['TR'] = max_
        except:
            print('err')
            row['TR'] = np.nan
#    df['ATR'] = df.TR.rolling(window=atr_window).mean()
#    
#    df['EMA'] = pd.ewma(df.Close, span=ema_window)
#    df['Upper'] = df.EMA + (std_coef * df.ATR)
#    df['Lower'] = df.EMA - (std_coef * df.ATR)
#    
#    df['Predicted_High'] = df.Upper.shift(predicted_periods)
#    df['Predicted_Avg'] = df.EMA.shift(predicted_periods)
#    df['Predicted_Low'] = df.Lower.shift(predicted_periods)
    
    #df.drop(['HL', 'HCl', 'LCl', 'TR', 'ATR', 'EMA', 'Upper', 'Lower'])
    
    return df


def calc_ichi(df, lookback_val):
    df['max_lp'] = df.High.rolling(lookback_val).max()
    df['min_lp'] = df.Low.rolling(lookback_val).min()
    df['avg_lp'] = (df.max_lp + df.min_lp)/2
    df['projected_val'] = df.avg_lp.shift(1)
    
    return df


def ichi_open_cross(df, span, percent_prof, exit_low_percent):
    # 1. Calculate the ichimoku projections for the next week
    df = df.reset_index()
    df = calc_ichi(df, span)
    
    # 2. Calculate the high of the next week
    df['shift_high'] = df.High.shift(-span+1)
    df['shift_low'] = df.Low.shift(-span+1)
    df['next_high'] = df.shift_high.rolling(span).max()
    df['next_low'] = df.shift_low.rolling(span).min()        
    
    
    # 3. Create a column for the previous close
    df['prev_close'] = df.Close.shift(1)
    
    # 4. Calculate the difference between the projected price and this week's high
    df['return_over_projected'] = (df.next_high - df.projected_val) / df.projected_val * 100
    df['loss_under_projected'] = (df.next_low - df.projected_val) / df.projected_val * 100
    
    # 5. Calculate return if sold at close
    df['close_of_week'] = df.Close.shift(-span)
    df['return_sold_close'] = (df.close_of_week - df.projected_val) / df.projected_val * 100
    
    # 6. Calculate whether price gets above a certain % of projected
    df['percent_sale_bool'] = df.return_over_projected > percent_prof
    df['half_percent_sale_bool'] = df.return_over_projected > percent_prof/2
    df['quarter_percent_sale_bool'] = df.return_over_projected > percent_prof/4
    
    # 7. Find where open low and reach sale happens
    df['buy_bool'] = (df.Open < df.projected_val) & (df.next_high >= df.projected_val)# & (df.prev_close > df.projected_val)
    df['exit_low_bool'] = df.loss_under_projected < exit_low_percent
    df['Loss_on_Low'] = (df.next_low - df.projected_val) / df.projected_val * 100
    
    
    df = calc_return(df, percent_prof, exit_low_percent, span)
            
    
    # . Find where the security opened the next week below the projected price and had a high above the projected price
    #df = df[df.rose_to_projected]  # df[(df.Open < df.projected_val) & (df.next_high >= df.projected_val)]
    
    return df


def calc_return(df, percent_prof, exit_low_percent, span):
    df['returns'] = 0.0
    for index, row in df.iterrows():
        if row.buy_bool:    #There is a buy
            full_high = row.projected_val * (1+(percent_prof/100))
            quarter_high = row.projected_val * (1+((percent_prof/4)/100))      
            #print('\nfull_high = {}'.format(full_high))
            #print('quarter_high = {}'.format(quarter_high))
            
            if (not row.half_percent_sale_bool) & (not row.exit_low_bool):
                #print('end of week profit in row {}'.format(index))
                df.loc[index, 'returns'] = row.return_sold_close
            elif (not row.half_percent_sale_bool) & (row.exit_low_bool):
                #print('exit low profit in row {}'.format(index))
                df.loc[index, 'returns'] = exit_low_percent
            else:   # It reached the half percent ladder
                # Create week long sub_df
                week_df = df.iloc[index:index+span,:]
                week_df = week_df.reset_index()
                crosses_half_idx = week_df.index[week_df.High > percent_prof/2].tolist()[0]
                week_after_cross_df = week_df.iloc[crosses_half_idx:, :]
                
#                print(np.max(week_after_cross_df.High))
#                print(np.min(week_after_cross_df.Low))
#                print(np.max(week_after_cross_df.High) > full_high)
#                print(np.min(week_after_cross_df.Low) < quarter_high)
                
                if (np.max(week_after_cross_df.High) > full_high) & (np.min(week_after_cross_df.Low) > quarter_high):   #Reaches full high and does not fall below quarter high
                    #print('full profit in row {}'.format(index))
                    df.loc[index, 'returns'] = percent_prof
                elif (np.max(week_after_cross_df.High) > full_high) &  (np.min(week_after_cross_df.Low) < (quarter_high)): # Reaches full high and also falls below quarter low
                    idx_full_high = week_after_cross_df.index[week_after_cross_df.High > full_high].tolist()[0]
                    idx_quarter_high = week_after_cross_df.index[week_after_cross_df.Low < quarter_high].tolist()[0]
                    #print('idx fh = {}'.format(idx_full_high))
                    #print('idx qh = {}'.format(idx_quarter_high))
                    if idx_full_high < idx_quarter_high:
                        print('full profit in row {}'.format(index))
                        df.loc[index, 'returns'] = percent_prof
                    elif idx_full_high > idx_quarter_high:
                        #print('quarter profit in row {}'.format(index))
                        df.loc[index, 'returns'] = percent_prof/4
                    else:
                        #print('quarter profit in row {}'.format(index))
                        df.loc[index, 'returns'] = percent_prof/4   
                elif (not (np.max(week_after_cross_df.High) > full_high)) &  (np.min(week_after_cross_df.Low) < (quarter_high)):
                    #print('quarter profit in row {}'.format(index))
                    df.loc[index, 'returns'] = percent_prof/4
                else:   # Reached half, but did not fall below a quarter or rise to full.  Sell out end of week
                    #print('end of week profits in row {}'.format(index))
                    df.loc[index, 'returns'] = row.return_sold_close
        else:
            df.loc[index, 'returns'] = np.nan

#        if row.percent_sale_bool & row.exit_low_bool & row.buy_bool:    # mixed case
#            current_date = row.DT
#            next_high = row.next_high
#            next_low = row.next_low
#            sub_df = df.iloc[index:index+span,:]
#            date_high = sub_df[sub_df.High == next_high].DT
#            date_low = sub_df[sub_df.Low == next_low].DT
#            time_to_high = (date_high - current_date).iloc[0].days
#            time_to_low = (date_low - current_date).iloc[0].days
#            
#            
#            if time_to_high < time_to_low:
#                df.iloc[index, -1] = percent_prof
#            elif time_to_high == time_to_low:
#                df.iloc[index, -1] = 0
#            else:
#                df.iloc[index, -1] = exit_low_percent
#                
#        elif row.percent_sale_bool & ~row.exit_low_bool & row.buy_bool:  # sell high case
#            #print('high')
#            df.iloc[index, -1] = percent_prof
#        
#        elif ~row.percent_sale_bool & row.exit_low_bool & row.buy_bool & ~row.quarter_percent_sale_bool :
#            #print('low')
#            df.iloc[index, -1] = exit_low_percent
        
#        elif ~row.percent_sale_bool & ~row.exit_low_bool & row.buy_bool & row.half_percent_sale_bool:
#            #print('close')
#            df.iloc[index, -1] = percent_prof/2
#        elif ~row.percent_sale_bool & ~row.exit_low_bool & row.buy_bool & ~row.half_percent_sale_bool & row.quarter_percent_sale_bool:
#            #print('close')
#            df.iloc[index, -1] = percent_prof/4
#        elif ~row.percent_sale_bool & ~row.exit_low_bool & row.buy_bool & ~row.half_percent_sale_bool & ~row.quarter_percent_sale_bool:
#            df.iloc[index, -1] = row.return_sold_close
            
    df['return_as_multiple'] = 1 + (df.returns / 100)
    df.return_as_multiple = df.return_as_multiple.replace(1.00, np.nan)
    
    return df

#if __name__ == '__main__':
#    main(send_email=False)
