#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:09:10 2022

Incorporated a rolling mad for key functions

@author: ellenyu

"""
from vqti.load import load_eod 
from vqti.profile import time_this 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from typing import List


@time_this 
def pandas_df_cci_rolling(df: pd.DataFrame, window: int=20) -> pd.DataFrame: 
	
    # Calculate the rolling average 
    sma = df.rolling(window).mean()
    #print(sma.head(21), '\n')
	
    # Calculate the rolling mean absolute deviation
	#df['mad'] = df['close'].rolling(window).apply(lambda x: pd.Series(x).mad()) #??? According to documentation .mad() takes a series, but the following works too
    mad = df.rolling(window).apply(lambda x: x.mad()) 
    assert type(mad) == pd.DataFrame, "Does not equal"
    #print(mad.head(21), '\n')

    # Put it all together using formula for cci 
    cci = (df - sma) / (0.015 * mad) # 0.015 is Lambert's constant 
    #print(cci.head(21), '\n')
    
    assert type(cci) == pd.DataFrame, "Output array is not same type as input array"
    assert len(cci) == len(df), "Output array is not same length as input array"
    
    return cci 


@time_this
def pandas_series_cci_rolling(series: pd.Series, window: int=20) -> pd.Series: 
	
    # Calculate the rolling average 
    sma: pd.Series = series.rolling(window).mean()
    #print(sma.head(21),'\n')
    
    # Calculate the rolling mean absolute deviation
    #mad = series.rolling(window).apply(lambda x: pd.Series(x).mad()) #??? According to documentation .mad() takes a series, but the following works too
    mad: pd.Series = series.rolling(window).apply(lambda x: x.mad())
    assert type(mad) == pd.Series, "Does not equal"
    #print(mad.head(21),'\n')

	# Put it all together using formula for cci
    cci: pd.Series = (series-sma) / (0.015 * mad)
    #print(cci.head(21),'\n')
    
    assert type(cci) == pd.Series, "Output array is not same type as input array"
    assert len(cci) == len(series), "Output array is not same length as input array"

    return cci


@time_this
def python_cci_loop(input_list: List[float], window: int=20) -> List[float]:
    
    cci = [None] * (window-1)
    
    for i in range(window, len(input_list)+1): 
        # Calculate the rolling average
        assert window == len(input_list[i-window:i]), "Lengths do not equal" # Double check window = len(input_list[i-window:i]) 
        window_mean = sum(input_list[i-window:i]) / window 
   
        # Calculate the rolling mean absolute deviation
        window_deviation = [x - window_mean for x in input_list[i-window:i]]
        window_abs_deviation = [abs(x) for x in window_deviation]
        assert window == len(window_abs_deviation), "Lengths do not equal" # Double check window = len(input_list[i-window:i]) 
        window_mad = sum(window_abs_deviation)/ window
    
        # Put it all together using formula for cci
        window_cci = (input_list[i-1]- window_mean) / (0.015 * window_mad)
        #print('cci:', window_cci, '\n')
        cci.append(window_cci)
    
    #print('cci:', cci, '\n')    
    
    assert type(cci) == list, "Output array is not same type as input array"
    assert len(cci) == len(input_list), "Output array is not same length as input array"
    
    return cci


@time_this
def numpy_cci_rolling(input_array: np.ndarray, window: int=20) -> np.ndarray:
    
    ## Using function to create a rolling window
    windows = np.lib.stride_tricks.sliding_window_view(input_array, window_shape=window)
    #print('input:', input_array, '\n')
    #print('window_array:', windows, '\n')
    
    # Calculate the rolling average
    sma = np.mean(windows, axis = 1)
    sma = np.pad(sma, (window-1, 0), 'constant', constant_values= np.nan)
    #print('sma:', sma, '\n')
	
    # Calculate the rolling mean absolute deviation
    window_deviation = windows - np.mean(windows, axis = 1, keepdims=True)
    #print('window_deviation:', window_deviation, '\n')
    window_abs_deviation = np.absolute(window_deviation)
    #print('window_abs_deviation:', window_abs_deviation, '\n')
    window_mad = np.mean(window_abs_deviation, axis =1)
    window_mad = np.pad(window_mad, (window-1, 0), 'constant', constant_values= np.nan)
    #print('window_mad:', window_mad, '\n')

    # Put it all together using formula for cci 
    cci = (input_array - sma) / (0.015 * window_mad) # 0.015 is Lambert's constant 
    #print('cci:', cci, '\n')
    
    assert type(cci) == np.ndarray, "Output array is not same type as input array"
    assert len(cci) == len(input_array), "Output array is not same length as input array"
 
    return cci 

    
if __name__ == '__main__':

#%%    
    # Building - test case for use with print statements while building
    
    ## Test data       
    test_list = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    
    window = 8
    
    pandas_df = pandas_df_cci_rolling(pd.DataFrame(test_list), window = window)
    pandas_series = pandas_series_cci_rolling(pd.Series(test_list), window = window)
    python = python_cci_loop(test_list, window = window)
    numpy = numpy_cci_rolling(np.array(test_list), window = window)
    
    print(pandas_df, '\n')
    print(pandas_series, 'n')
    print(python, '\n')
    print(numpy, '\n')
    
#%%
    # Testing - checking methods against simple truth case 
    
    # window = 2 case
    truth_list = [None, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667]

    window = 2
    pandas_df = pandas_df_cci_rolling(pd.DataFrame(test_list), window = window)
    pandas_series = pandas_series_cci_rolling(pd.Series(test_list), window = window)
    python = python_cci_loop(test_list, window = window)
    numpy = numpy_cci_rolling(np.array(test_list), window = window)
    
    print('Testing pandas_df results against truth_list:')
    precision = 64 
    for item in range(len(truth_list)):
        #print(item)
        if (truth_list[item] is None):
            #print (truth_list[item])
            continue 
        else:     
            np.testing.assert_equal(round(pandas_df.values.tolist()[item][0], precision), round(truth_list[item], precision)), "Does not equal"
    print('Test passed!\n') 
    
    print('Testing pandas_series results against truth_list:')
    precision = 64 
    for item in range(len(truth_list)):
        #print(item)
        if (truth_list[item] is None):
            #print (truth_list[item])
            continue 
        else:     
            np.testing.assert_equal(round(pandas_series.tolist()[item], precision), round(truth_list[item], precision)), "Does not equal"
    print('Test passed!\n') 
    
    print('Testing python results against truth_list:')
    precision = 64 
    for item in range(len(truth_list)):
        #print(item)
        if (truth_list[item] is None):
            #print (truth_list[item])
            continue 
        else:     
            np.testing.assert_equal(round(python[item], precision), round(truth_list[item], precision)), "Does not equal"
    print('Test passed!\n') 
    
    print('Testing numpy results against truth_list:')
    precision = 64 
    for item in range(len(truth_list)):
        #print(item)
        if (truth_list[item] is None):
            #print (truth_list[item])
            continue 
        else:     
            np.testing.assert_equal(round(numpy[item], precision), round(truth_list[item], precision)), "Does not equal"
    print('Test passed!\n') 

    
#%%     
    
    # Testing - printing results on df to manually check match 
    df = load_eod('AWU')
    #print(df.head())
    
    window = 5
    
    #pandas_df_cci_rolling(df, window = window) # commented out because it takes long to run 
    pandas_series = pandas_series_cci_rolling(df.close, window = window)
    python = python_cci_loop(df.close.tolist(), window = window)
    numpy = numpy_cci_rolling(df.close.to_numpy(), window = window)
    
    print(pandas_series[:20], '\n')
    print(python[:20], '\n')
    print(numpy[:20], '\n')
#%%
    
    # Testing - checking if methods on df produces the same results 
    # Testing python methods vs. pandas_series method 
    print('Testing python vs. pandas_series results:')
    precision = 9 # Passes at precision = 9 
    for item in range(len(python)):
        #print(item)
        if (python[item] is None):
            print (python[item])
        else:     
            np.testing.assert_equal(round(python[item],precision), round(pandas_series.tolist()[item], precision)), "Does not equal"
            #assert python[item] == pandas_series.tolist()[item], "Does not equal"
    print('Passes!\n')   
     
    # Testing python methods vs. numpy method
    print('Testing python vs. numpy results:')
    precision = 64 # Passes at precision = 64
    for item in range(len(python)):
        #print(item)
        if (python[item] is None):
            print (python[item])
        else:     
            np.testing.assert_equal(round(python[item],precision), round(numpy.tolist()[item], precision)), "Does not equal"
            #assert python[item] == pandas_series.tolist()[item], "Does not equal"
    print('Passes!\n')
    
    