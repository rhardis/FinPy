# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:30:30 2019

@author: Richard Hardis
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def apply_pca(inputs, outputs):
    #inputs is a pandas dataframe
    #outputs is a pandas series
    
    train_data, test_data, train_lbl, test_lbl = train_test_split(inputs, outputs, test_size=9/10, random_state=0)
    
    scaler = StandardScaler()
    scaler.fit(train_data)
    
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
    pca = PCA(0.95)
    
    pca.fit(train_data)
    
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    
    return df


def get_best_vars_percentile(X, y, percentile):
    X_new = SelectPercentile(chi2, percentile=percentile).fit_transform(X.values, y.values)
    return X_new


def normalize_df(X):
    for col in X.columns:
        xmin = np.min(X[col])
        xmax = np.max(X[col])
        X[col] = (X[col] - xmin) / (xmax - xmin)

    return X
    
    
    