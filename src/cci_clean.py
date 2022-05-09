#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:37:36 2022

Commodity Channel index 

@author: ellenyu
"""
# Import packages 
from vqti.load import load_eod 
from vqti.profile import time_this
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

@time_this
def compute_cci(df: pd.DataFrame, n_days: int=20) -> pd.DataFrame: 
    # Define the typical price for a day 
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3 
    
    # Find the moving average of the typical prices
    df['sma'] = df['tp'].rolling(n_days).mean()

    # Calculate the mean average divergence
    df['mad'] = df['tp'].rolling(n_days).apply(lambda x: pd.Series(x).mad())

    # Put it all together using formula for cci
    df['cci'] = (df['tp'] - df['sma']) / (0.015 * df['mad']) #Lambert (creator of CCI) recommends 0.015 for stability.
    
    return df

@time_this
def compute_cci_v2(df: pd.DataFrame, n_days: int=20) -> pd.DataFrame: 
    # Vectorization implementation for typical price
    arr_high = df["high"].array 
    arr_low = df["low"].array
    arr_close = df["close"].array
    arr_tp = []
    for i in range(len(arr_high)):
        arr_tp.append((arr_high[i] + arr_low[i] + arr_close[i])/3)
    df['tp'] =arr_tp 
    
    # Find the moving average of the typical prices
    df['sma'] = df['tp'].rolling(n_days).mean()

    # Calculate the mean average divergence
    df['mad'] = df['tp'].rolling(n_days).apply(lambda x: pd.Series(x).mad())

    # Put it all together using formula for cci
    df['cci'] = (df['tp'] - df['sma']) / (0.015 * df['mad']) #Lambert (creator of CCI) recommends 0.015 for stability.
    
    return df

def graph_cci(df:pd.DataFrame, n_days: int=14):
    # Plot the price series chart and cci 
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(2, 1, 1)
    #ax.set_xticklabels([])
    plt.plot(df['close'],lw=1)
    plt.title('Price Chart')
    plt.ylabel('Close Price')
    plt.grid(True)
    
    bx = fig.add_subplot(2, 1, 2)
    plt.plot(compute_cci(df)["cci"],lw=0.75,linestyle='-',label='CCI')
    plt.title('CCI Values')
    
    plt.legend(loc=2,prop={'size':9.5})
    plt.ylabel('CCI values')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
	   
    # Load and view data 
    df = load_eod('AWU')
    #print(df.head())   

    # View results
    method_1 = compute_cci(df, n_days=20)
    method_2 = compute_cci_v2(df, n_days=20)

    pd.set_option('display.max_columns', None)
    print(method_1.head(20))
    pd.reset_option('display.max_columns')
    
    graph_cci(df, n_days=20)


