# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:25:27 2019

@author: Richard Hardis
"""

import os
import numpy as np
import pandas as pd

df = pd.read_csv(os.path.join(r'C:\Users\Richard Hardis\Documents\GitHub\FinPy\march-strategy', 'returns_summary.csv'))
data = df['return']

stdev = np.std(data)
mean = np.mean(data)
pos_ret_count = len([i for i in data if i > 0])
