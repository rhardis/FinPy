# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:55:12 2019

@author: Richard Hardis
"""
import numpy as np
import pandas as pd



def macd_stochastic(macd_arg_dict, stoch_arg_dict):

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
            
    return signal_series