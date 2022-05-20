#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:09:10 2022

[5-17-22] Incorporated a rolling mad for key functions

[5-19-22] Created modified versions where we took out Lambert's constant

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


#@time_this
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


#@time_this
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
        if window_mad == 0:
            print('i:', i, ',', 'window_mad:', window_mad, '\n')
            
        # Put it all together using formula for cci
        ## implement try except block to handle when window mad = 0 [5-20-22]
        try: 
            window_cci = (input_list[i-1]- window_mean) / (0.015 * window_mad)
        except ZeroDivisionError:
            window_cci = 0
        
        #print('cci:', window_cci, '\n')
        cci.append(window_cci)
    
    #print('cci:', cci, '\n')    
    
    assert type(cci) == list, "Output array is not same type as input array"
    assert len(cci) == len(input_list), "Output array is not same length as input array"
    
    return cci


#@time_this
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


#@time_this 
def pandas_df_ccimodified_rolling(df: pd.DataFrame, window: int=20) -> pd.DataFrame: 
	
    # Calculate the rolling average 
    sma = df.rolling(window).mean()
    #print(sma.head(21), '\n')
	
    # Calculate the rolling mean absolute deviation
	#df['mad'] = df['close'].rolling(window).apply(lambda x: pd.Series(x).mad()) #??? According to documentation .mad() takes a series, but the following works too
    mad = df.rolling(window).apply(lambda x: x.mad()) 
    assert type(mad) == pd.DataFrame, "Does not equal"
    #print(mad.head(21), '\n')

    # Put it all together using formula for cci 
    cci = (df - sma) / (mad) # 0.015 is Lambert's constant 
    #print(cci.head(21), '\n')
    
    assert type(cci) == pd.DataFrame, "Output array is not same type as input array"
    assert len(cci) == len(df), "Output array is not same length as input array"
    
    return cci 


#@time_this
def pandas_series_ccimodified_rolling(series: pd.Series, window: int=20) -> pd.Series: 
	
    # Calculate the rolling average 
    sma: pd.Series = series.rolling(window).mean()
    #print(sma.head(21),'\n')
    
    # Calculate the rolling mean absolute deviation
    #mad = series.rolling(window).apply(lambda x: pd.Series(x).mad()) #??? According to documentation .mad() takes a series, but the following works too
    mad: pd.Series = series.rolling(window).apply(lambda x: x.mad())
    assert type(mad) == pd.Series, "Does not equal"
    #print(mad.head(21),'\n')

	# Put it all together using formula for cci
    cci: pd.Series = (series-sma) / (mad)
    #print(cci.head(21),'\n')
    
    assert type(cci) == pd.Series, "Output array is not same type as input array"
    assert len(cci) == len(series), "Output array is not same length as input array"

    return cci


#@time_this
def python_ccimodified_loop(input_list: List[float], window: int=20) -> List[float]:
    
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
        if window_mad == 0:
            print('i:', i, ',', 'window_mad:', window_mad, '\n')
    
        # Put it all together using formula for cci
        ## implement try except block to handle when window mad = 0 [5-20-22]
        try: 
            window_cci = (input_list[i-1]- window_mean) / (window_mad)
        except ZeroDivisionError:
            window_cci = 0
        
        #print('cci:', window_cci, '\n')
        cci.append(window_cci)
    
    #print('cci:', cci, '\n')    
    
    assert type(cci) == list, "Output array is not same type as input array"
    assert len(cci) == len(input_list), "Output array is not same length as input array"
    
    return cci


#@time_this
def numpy_ccimodified_rolling(input_array: np.ndarray, window: int=20) -> np.ndarray:
    
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
    ## implement try except block to handle when window mad = 0 [5-20-22]
    cci = (input_array - sma) / (window_mad) # 0.015 is Lambert's constant 
    
    #print('cci:', cci, '\n')
    
    assert type(cci) == np.ndarray, "Output array is not same type as input array"
    assert len(cci) == len(input_array), "Output array is not same length as input array"
 
    return cci 

    
if __name__ == '__main__':

#%%    
    # Building - test case for use with print statements while building
    
    ## Test data       
    #test_list = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    test_list = [1., 1., 1., 4., 5., 6., 7., 8., 9., 10.] # Note, returns ZeroDivisionError: float division by zero for e.g. window = 3 [5-20-22]
    
    window = 3
    
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
    test_list = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
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
    
    #print(pandas_series[:20], '\n')
    #print(python[:20], '\n')
    #print(numpy[:20], '\n')
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
    
    #%% 
    
    # Extent of testing the modified versions - printing out results [5-19-22]
        
    ## Test data       
    test_list = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    
    window = 8
    
    pandas_df = pandas_df_cci_rolling(pd.DataFrame(test_list), window = window)
    pandas_series = pandas_series_cci_rolling(pd.Series(test_list), window = window)
    python = python_cci_loop(test_list, window = window)
    numpy = numpy_cci_rolling(np.array(test_list), window = window)
    
    pandas_df_modified = pandas_df_ccimodified_rolling(pd.DataFrame(test_list), window = window)
    pandas_series_modified = pandas_series_ccimodified_rolling(pd.Series(test_list), window = window)
    python_modified = python_ccimodified_loop(test_list, window = window)
    numpy_modified = numpy_ccimodified_rolling(np.array(test_list), window = window)
    
    print(pandas_df, '\n')
    print(pandas_series, 'n')
    print(python, '\n')
    print(numpy, '\n')
    
    print(pandas_df_modified, '\n')
    print(pandas_series_modified, 'n')
    print(python_modified, '\n')
    print(numpy_modified, '\n')
    
    # Observations 
    # cci is ~116.67 in Lambert's version; cci is ~1.75 in modified version 
    # when you divide by .015, it is the same thing as multiplying by 1/.015 aka ~66.67 
    # so, Lambert's version is scaling everything ~6.67 - I wonder why he chose to do that
    # Ans: Lambert set the constant at . 015 to ensure that approximately 70 to 80 
    # percent of CCI values would fall between -100 and +100. This percentage 
    # also depends on the look-back period. A shorter CCI (10 periods) will be 
    # more volatile with a smaller percentage of values between +100 and -100

    