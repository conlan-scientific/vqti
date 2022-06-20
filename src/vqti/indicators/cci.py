#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:39:28 2022

cci functions for sharing 

@author: ellenyu

"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from typing import List


def pandas_cci_rolling(series: pd.Series, window: int=20, factor: int=1) -> pd.Series: 
	
    # Do assert statements take time? Double check this. 
    assert window > 0, "window must be greater than zero"
    assert factor > 0, "factor must be greater than zero"
    
    # Calculate the rolling average 
    sma: pd.Series = series.rolling(window).mean()
    #print('sma:', sma.head(21),'\n')
    
    # Calculate the rolling mean absolute deviation
    #mad = series.rolling(window).apply(lambda x: pd.Series(x).mad()) #??? According to documentation .mad() takes a series, but the following works too
    mad: pd.Series = series.rolling(window).apply(lambda x: x.mad())
    assert type(mad) == pd.Series, "Does not equal"
    #print('window_mad:', mad.head(21),'\n')

	# Put it all together using formula for cci
    cci: pd.Series = (series-sma) / (factor * mad) # Note, Lambert's constant is 0.015
    #print('cci:', cci.head(21),'\n')
    
    assert type(cci) == pd.Series, "Output array is not same type as input array"
    assert len(cci) == len(series), "Output array is not same length as input array"

    return cci


def python_cci_loop(input_list: List[float], window: int=20, factor: int=1) -> List[float]:
    
    # Do assert statements take time? Double check this. 
    assert window > 0, "window must be greater than zero"
    assert factor > 0, "factor must be greater than zero"
    
    cci = [None] * (window-1)
    
    for i in range(window, len(input_list)+1): 
        # Calculate the rolling average
        assert window == len(input_list[i-window:i]), "Lengths do not equal" # Double check window = len(input_list[i-window:i]) 
        window_mean = sum(input_list[i-window:i]) / window 
        #print('sma:', window_mean, '\n')
   
        # Calculate the rolling mean absolute deviation
        window_deviation = [x - window_mean for x in input_list[i-window:i]]
        window_abs_deviation = [abs(x) for x in window_deviation]
        assert window == len(window_abs_deviation), "Lengths do not equal" # Double check window = len(input_list[i-window:i]) 
        window_mad = sum(window_abs_deviation)/ window
        #print('window_mad:', window_mad, '\n')
        if window_mad == 0:
            print('i:', i, ',', 'window_mad:', window_mad, '\n')
            
        # Put it all together using formula for cci
        ## implement try except block to handle when window mad = 0 [5-20-22]
        try: 
            window_cci = (input_list[i-1]- window_mean) / (factor * window_mad) # Note, Lambert's constant is 0.015
        except ZeroDivisionError:
            window_cci = 0 # Should cc return 0 if mad is 0? Double check this 
            
        #print('cci:', window_cci, '\n')
        cci.append(window_cci)
    
    #print('cci:', cci, '\n')    
    
    assert type(cci) == list, "Output array is not same type as input array"
    assert len(cci) == len(input_list), "Output array is not same length as input array"
    
    return cci


def numpy_cci_rolling(input_array: np.ndarray, window: int=20, factor: int=1) -> np.ndarray:
    
    # Do assert statements take time? Double check this. 
    assert window > 0, "window must be greater than zero"
    assert factor > 0, "factor must be greater than zero"
    
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
    cci = (input_array - sma) / (factor * window_mad) # Note, Lambert's constant is 0.015    
    #print('cci:', cci, '\n')
    
    assert type(cci) == np.ndarray, "Output array is not same type as input array"
    assert len(cci) == len(input_array), "Output array is not same length as input array"
 
    return cci 

    
if __name__ == '__main__':
    
    # Testing - checking while I'm building and checking for cci boundaries
    #test_list = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    test_list = [10., 9., 8., 7., 6., 5., 4., 3., 2., 1.]
    #test_list = [10., 9.5, 9., 8.5, 8., 7.5, 7., 3., 2., 1.] # Trying to push cci to -2
    #test_list = [10., 9.5, 9., 8.5, 8., 7.5, 7., 1., 1., 1.] # Trying to push cci to -3
    #test_list = [10., 10., 10., 10., 10., 10., 10., 9., 1., 1.1] # Trying to push cci to -4
    
    window = 8
    factor = 1

    pandas = pandas_cci_rolling(pd.Series(test_list), window=window, factor=factor)
    python = python_cci_loop(test_list, window=window, factor=factor)
    numpy = numpy_cci_rolling(np.array(test_list), window=window, factor=factor) 
    
    print(pandas)
    print(python)
    print(numpy)
    
    # Observations
    # If these values are prices, then prices have fallen steadily. Note, even though 
    # prices have fallen to one-tenth of its values, it has not trigger a buy event which 
    # I've set to default to -3. I guess that's what we want... we want to buy if there
    # is an inordinate decline and not a steady decline...
    
#%%    

    # Testing - checking methods against simple truth case 
 
    # window = 2, factor = 0.015 case
    test_list = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    truth_list = [None, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667]

    window = 2
    factor = 0.015
    numpy = numpy_cci_rolling(np.array(test_list), window=window, factor=factor)
    python = python_cci_loop(test_list, window=window, factor=factor)
    pandas = pandas_cci_rolling(pd.Series(test_list), window=window, factor=factor)
    

    print('Testing pandas results against truth_list:')
    precision = 64 
    for item in range(len(truth_list)):
        if (truth_list[item] is None):
            continue 
        else:     
            np.testing.assert_equal(round(pandas.tolist()[item], precision), round(truth_list[item], precision)), "Does not equal"
    print('Test passed!\n') 
    
    print('Testing python results against truth_list:')
    precision = 64 
    for item in range(len(truth_list)):
        if (truth_list[item] is None):
            continue 
        else:     
            np.testing.assert_equal(round(python[item], precision), round(truth_list[item], precision)), "Does not equal"
    print('Test passed!\n') 
    
    print('Testing numpy results against truth_list:')
    precision = 64 
    for item in range(len(truth_list)):
        if (truth_list[item] is None):
            continue 
        else:     
            np.testing.assert_equal(round(numpy[item], precision), round(truth_list[item], precision)), "Does not equal"
    print('Test passed!\n') 

    
    # # Observations 
    # # cci is ~116.67 in Lambert's version; cci is ~1.75 in modified version 
    # # when you divide by .015, it is the same thing as multiplying by 1/.015 aka ~66.67 
    # # so, Lambert's version is scaling everything ~6.67 - I wonder why he chose to do that
    # # Ans: Lambert set the constant at . 015 to ensure that approximately 70 to 80 
    # # percent of CCI values would fall between -100 and +100. This percentage 
    # # also depends on the look-back period. A shorter CCI (10 periods) will be 
    # # more volatile with a smaller percentage of values between +100 and -100

