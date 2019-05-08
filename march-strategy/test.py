# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:55:14 2019

@author: Richard Hardis
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2

X, y = load_digits(return_X_y=True)
X = pd.DataFrame(X)
X.shape
print(type(X))
y.shape
print(type(y))
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
X_new.shape
print(type(X))