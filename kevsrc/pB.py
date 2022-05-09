import sys
sys.path.append('..\src')
import math
from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot
import pandas as pd
import numpy as np
from typing import List

"""
Simple Moving Average
"""

def python_sma_k (values: List[float], n: int = 2, m: int = 2):

    values = load_eod('AWU') # loads the dataframe in question
    close = values.loc[:,'close'] # loads only the values in the close column

    # simple moving average formula
    amov = [0] * (n-1) # initializes the moving average formula
    rolling_total = sum(close[:n]) # adds up numbers associated with the moving average range
    amov.append(rolling_total/n)
    
    for k in range(n, len(close)):  #loops moving average formula through column range
        rolling_total = rolling_total - close[k-n]
        rolling_total = rolling_total + close[k]
        amov.append(rolling_total/n)
        
        if amov[0] == 0:
            del amov[0] # truncates 0 values off of moving average list
    
        
    N = 2 #multiplier for the standard deviation
    stddev_k = [0] * (n-1) # initializes the standard deviation formula
        
    for j in range(m, len(close)): #a loop to append values to a standard deviation list
        dev = (amov[j] - amov[m])**2
        var = sum(dev)/m
        stddev = math.sqrt(var)
        stddev_k.append(N*stddev)
    
        if stddev_k[0] == 0:
            del stddev_k[0] # truncates 0 values off of standard deviation list
    
    return stddev_k



if __name__=="__main__":    
    
    df = load_eod('AWU')
    print (df)
    
    amov_result = python_sma_k(df)
    print (amov_result)