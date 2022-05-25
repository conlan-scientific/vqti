#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 12:04:12 2022

The following are rolling sum aka moving sum functions that can be adapted to become 
moving average functions or pared down to be rolling window functions. 

Pending: 
    * Program something to handle when users enter window = 0 or window = negative number
    * Address the work arounds so, I can extract rolling functions from rollingsum functions
    * Create a numbas implementation and negative test cases 

@author: ellenyu

"""
from vqti.load import load_eod 
from vqti.profile import time_this 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from typing import List
from numba import jit


@time_this 
def pandas_df_rollingSum_loop(df:pd.DataFrame, window:int=20) -> pd.DataFrame():
    
    # Find the rolling sum by
    
    # ## Using a for loop
    rollingsum = pd.DataFrame(index=df.index[:window-1],columns=df.columns)
    #print('input:', df, '\n')
    #print('df of nans:', rollingsum, '\n')
    for i in range(window, len(df)+1): 
        #window = df[i-window:i] # Raises TypeError: cannot do slice indexing on RangeIndex with these indexers
        #print(window, '\n')
        window_sum = df[i-window:i].sum().astype(float) # Work around 
        #print(window_sum, '\n')
        rollingsum = rollingsum.append(window_sum, ignore_index=True)
    #print('rollingsum:', rollingsum, '\n')
    
    assert type(rollingsum) == pd.DataFrame, "Output array is not same type as input array"
    assert len(rollingsum) == len(df), "Output array is not same length as input array"
    
    return rollingsum


@time_this 
def pandas_df_rollingSum_cumsum(df:pd.DataFrame, window:int=20) -> pd.DataFrame:
 	
    # Find the rolling sum by 
    
    ## Calculating a cumsum
    accum = df.cumsum()
    #print('input:', df, '\n')
    #print('accum:', accum, '\n')
    
    ## Subtracting a cumsum
    subtract_cumsum = accum.shift(window)
    subtract_cumsum.iloc[window-1] = 0
    rollingsum = accum - subtract_cumsum
    #print('subtract this:', subtract_cumsum, '\n')
    #print('rollingsum:', rollingsum, '\n')
    
    assert type(rollingsum) == pd.DataFrame, "Output array is not same type as input array"
    assert len(rollingsum) == len(df), "Output array is not same length as input array"
    
    return rollingsum


@time_this 
def pandas_df_rollingSum_rolling (df:pd.DataFrame, window:int=20) -> pd.DataFrame: 
 	
    # Find the rolling sum by 
    
    ## Using function to create a rolling window and sum results
    rollingsum: pd.DataFrame = df.rolling(window).sum()
    #print('rollingsum:', rollingsum, '\n')

    
    assert type(rollingsum) == pd.DataFrame, "Output array is not same type as input array"
    assert len(rollingsum) == len(df), "Output array is not same length as input array"
    
    return rollingsum


@time_this 
def pandas_series_rollingSum_loop(series:pd.Series, window:int=20) -> pd.Series:
   
    # Find the rolling sum by
    
    # ## Using a for loop
    rollingsum = pd.Series(index=series.index[:window-1], dtype = float)
    #print('input:', series, '\n')
    #print('df of nans:', rollingsum, '\n')
    for i in range(window, len(series)+1): 
        #print(i, '\n')
        #window = series[i-window:i] #TypeError: cannot do slice indexing on RangeIndex with these indexers [0    1.0 dtype: float64] of type Series
        #print(type(window))
        #print(window, '\n')
        window_sum = series[i-window:i].sum() # Work around
        #print(type(window_sum))
        #print(window_sum, '\n')
        #rollingsum.at[i] = window_sum # # This method raised TypeError: value should be a 'Timestamp' or 'NaT'. Got 'int' instead.
        #rollingsum.iloc[i] = window_sum # This method raised IndexError: iloc cannot enlarge its target object
        rollingsum = rollingsum.append(pd.Series(window_sum), ignore_index=True) 
        
    #print('rollingsum:', rollingsum, '\n')
    
    assert type(rollingsum) == pd.Series, "Output array is not same type as input array"
    assert len(rollingsum) == len(series), "Output array is not same length as input array"
    
    return rollingsum

    
@time_this 
def pandas_series_rollingSum_cumsum(series:pd.Series, window:int=20) -> pd.Series: 
 	
    # Find the rolling sum by 
    
    ## Calculating a cumsum
    accum = series.cumsum()
    #print('input:', series, '\n')
    #print('accum:', accum, '\n')
    
    ## Subtracting a cumsum
    subtract_cumsum = accum.shift(window)
    subtract_cumsum.iloc[window-1] = 0
    rollingsum = accum - subtract_cumsum
    #print('subtract this:', subtract_cumsum, '\n')
    #print('rollingsum:', rollingsum, '\n')
    
    assert type(rollingsum) == pd.Series, "Output array is not same type as input array"
    assert len(rollingsum) == len(series), "Output array is not same length as input array"
    
    return rollingsum


#@time_this 
def pandas_series_rollingSum_rolling (series:pd.Series, window:int=20) -> pd.Series: 
 	
    # Find the rolling sum by 
    
    ## Using function to create a rolling window and sum results
    rollingsum: pd.Series = series.rolling(window).sum()
    #print('rollingsum:', rollingsum, '\n')

    
    assert type(rollingsum) == pd.Series, "Output array is not same type as input array"
    assert len(rollingsum) == len(series), "Output array is not same length as input array"
    
    return rollingsum


@time_this
def python_rollingSum_cumsum (input_list: List[float], window: int=20) -> List[float]:
    
    # Find the rolling sum by 
    
    ## Calculating a cumsum
    accum = []
    #print('input:', input_list, '\n')
    for i in range(1, len(input_list)+1):
        cumSum = sum(input_list[0:i:1]) # This is extended slice notation of [start: end: step]
        accum.append(cumSum) # To add int to list, use append. Reminder, lists are mutable 
    #print('cumulative sum:', accum, '\n')
    
    ## Subtracting a cumsum
    relevant_cumsum = accum[window-1:]
    subtract_cumsum = accum[:-window]
    subtract_cumsum = [0] + subtract_cumsum # To add list to list, can use + operator 
    rollingsum = [element1 - element2 for (element1, element2) in zip(relevant_cumsum, subtract_cumsum)] # To elementwise subtract list from list, use list comprehension and zip
    rollingsum = [None]*(window-1) + rollingsum
    #print('relevant cumsum:', relevant_cumsum, '\n')
    #print('subtract this:', subtract_cumsum, '\n')
    #print('rollingsum:', rollingsum, '\n')
    
    assert type(rollingsum) == list, "Output array is not same type as input array"
    assert len(rollingsum) == len(input_list), "Output array is not same length as input array"
    
    return rollingsum


@time_this
def python_rollingSum_loop (input_list: List[float], window: int=20) -> List[float]:
    
    # Find the rolling sum by
    
    ## Using a for loop
    rollingsum = [None] * (window-1)
    #print('input:', input_list, '\n')
    #print('array of nans:', rollingsum, '\n')
    for i in range(window, len(input_list)+1): 
        #window = input_list[i-window:i] # Raises TypeError: unsupported operand type(s) for -: 'int' and 'list'
        window_sum = sum(input_list[i-window:i]) # Work around 
        rollingsum.append(window_sum)
    #print('rollingsum:', rollingsum, '\n')
    
    assert type(rollingsum) == list, "Output array is not same type as input array"
    assert len(rollingsum) == len(input_list), "Output array is not same length as input array"
    
    return rollingsum


#@time_this
def python_rollingSum_loop_v2 (input_list: List[float], window: int=20) -> List[float]:
    
    # Find the rolling sum by
    
    ## Using a for loop [Chris implementation]
    rollingsum = [None] * (window-1)
    #print(rollingsum, '\n')
    accum = sum(input_list[:window])
    #print('first rollingsum:', accum, '\n')
    rollingsum.append(accum)
    #print(rollingsum, '\n')
    for i in range(window, len(input_list)):
        accum += input_list[i] # Add one index
        accum -= input_list[i-window] # Subtract one index so that it is always window number of indices
        rollingsum.append(accum)
    #print('rollingsum:', rollingsum,'\n') 
    
    assert type(rollingsum) == list, "Output array is not same type as input array"
    assert len(rollingsum) == len(input_list), "Output array is not same length as input array"

    return rollingsum

        
@time_this
def numpy_rollingSum_cumsum (input_array: np.ndarray, window: int=20) -> np.ndarray:
    
    # Find the moving average of closing price
    accum = input_array.cumsum()
    #print('input:', input_array, '\n')
    #print('accum:', accum, '\n')

    # Find the rolling sum by 
    
    ## Calculating a cumsum 
    accum = input_array.cumsum()
    #print('input:', input_array, '\n')
    #print('accum:', accum, '\n')

    ## Subtracting a cumsum
    relevant_cumsum = accum[window-1:]
    relevant_cumsum = np.pad(relevant_cumsum, (window-1, 0), 'constant', constant_values= np.nan)
    subtract_cumsum = accum[:-window]
    subtract_cumsum = np.pad(subtract_cumsum, (1, 0), 'constant', constant_values= 0) # This line of code is here because a number minus a nan results in a nan
    subtract_cumsum = np.pad(subtract_cumsum, (window-1, 0), 'constant', constant_values= np.nan)
    rollingsum = relevant_cumsum - subtract_cumsum
    #print('relevant cumsum:', relevant_cumsum, '\n')
    #print('subtract this:', subtract_cumsum, '\n')
    #print('rollingsum:', rollingsum, '\n')
    
    assert type(rollingsum) == np.ndarray, "Output array is not same type as input array"
    assert len(rollingsum) == len(input_array), "Output array is not same length as input array"
    
    return rollingsum


@time_this
def numpy_rollingSum_loop (input_array: np.ndarray, window: int=20) -> np.ndarray:
    
    # Find the rolling sum by
    
    ## Using a for loop
    rollingsum = np.empty(window-1) # Create an array with specified shape 
    rollingsum[:] = np.nan # Fill the array with nans
    #print('input:', input_array, '\n')
    #print('array of nans:', rollingsum, '\n')
    for i in range(window, len(input_array)+1): 
        #window = input_array[i-window:i] # Raises TypeError: only integer scalar arrays can be converted to a scalar index
        window_sum = np.sum(input_array[i-window:i]) # Work around
        rollingsum = np.append(rollingsum, window_sum)
    #print('rollingsum:', rollingsum, '\n')
    
    assert type(rollingsum) == np.ndarray, "Output array is not same type as input array"
    assert len(rollingsum) == len(input_array), "Output array is not same length as input array"
    
    return rollingsum

        
@time_this
def numpy_rollingSum_rolling (input_array: np.ndarray, window: int=20) -> np.ndarray:
    
    # Fing the rolling sum by 
    
    ## Using function to create a rolling window
    windows = np.lib.stride_tricks.sliding_window_view(input_array, window_shape=window)
    #print('input:', input_array, '\n')
    #print('windows:' , windows, '\n')
    
    ## Sum values in the rolling window
    rollingsum: np.ndarray = np.sum(windows, axis=1)
    #print('sum over axis = None:', np.sum(windows), '\n')
    #print('sum over axis = 0:', np.sum(windows, axis=0), '\n')
    #print('sum over axis = 1:', np.sum(windows, axis=1), '\n')
    rollingsum = np.pad(rollingsum, (window-1, 0), 'constant', constant_values= np.nan)
    #print(rollingsum)
    
    assert type(rollingsum) == np.ndarray, "Output array is not same type as input array"
    assert len(rollingsum) == len(input_array), "Output array is not same length as input array"
    
    return rollingsum


if __name__ == '__main__':

#%% 
    # # Building - test case for use wiht print statements while building
    
    # ## Test data       
    # test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # window = 5
    
    # pandas_df_rollingSum_loop(pd.DataFrame(test_list), window= window)
    # pandas_df_rollingSum_cumsum(pd.DataFrame(test_list), window= window)
    # pandas_df_rollingSum_rolling(pd.DataFrame(test_list), window= window)
   
    # pandas_series_rollingSum_loop(pd.Series(test_list), window= window)
    # pandas_series_rollingSum_cumsum(pd.Series(test_list), window= window)
    # pandas_series_rollingSum_rolling(pd.Series(test_list), window= window)
    
    # numpy_rollingSum_cumsum(np.array(test_list, dtype=float), window= window)
    # numpy_rollingSum_loop(np.array(test_list, dtype=float), window= window)
    # numpy_rollingSum_rolling(np.array(test_list, dtype=float), window= window)
    
    # python_rollingSum_cumsum(test_list, window= window)
    # python_rollingSum_loop(test_list, window= window)
    # python_rollingSum_loop_v2(test_list, window= window)

#%% 
    # # Testing - testing that functions run on dataframe data. Yet to test validity of results
    
    # ## Load data
    # df = load_eod('AWU')
    # print(df.head(5))
    
    # pandas_df_rollingSum_loop(df, window=3)
    # pandas_df_rollingSum_cumsum(df, window=3)
    # pandas_df_rollingSum_rolling(df, window=3)
    
    # pandas_series_rollingSum_loop(df.close, window=3)
    # pandas_series_rollingSum_cumsum(df.close, window=3)
    # pandas_series_rollingSum_rolling(df.close, window=3)
    
    # numpy_rollingSum_cumsum(df.close.to_numpy(), window=3)
    # numpy_rollingSum_loop(df.close.to_numpy(), window=3)
    # numpy_rollingSum_rolling(df.close.to_numpy(), window=3)
    
    # python_rollingSum_cumsum(df.close.tolist(), window=3)
    # python_rollingSum_loop(df.close.tolist(), window=3)
    # python_rollingSum_loop_v2(df.close.tolist(), window=3)

#%%
    # # Testing - testing window = 1 through length of test case. Yet to address what happens when window = 0
    
    # ## Test data       
    # test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # for window in range (1, len(test_list)+1):
        
    #     print('\nTesting window=', window)

    #     pandas_df_rollingSum_loop(pd.DataFrame(test_list), window= window)
    #     pandas_df_rollingSum_cumsum(pd.DataFrame(test_list), window= window)
    #     pandas_df_rollingSum_rolling(pd.DataFrame(test_list), window= window)
       
    #     pandas_series_rollingSum_loop(pd.Series(test_list), window= window)
    #     pandas_series_rollingSum_cumsum(pd.Series(test_list), window= window)
    #     pandas_series_rollingSum_rolling(pd.Series(test_list), window= window)
        
    #     numpy_rollingSum_cumsum(np.array(test_list, dtype=float), window= window)
    #     numpy_rollingSum_loop(np.array(test_list, dtype=float), window= window)
    #     numpy_rollingSum_rolling(np.array(test_list, dtype=float), window= window)
        
    #     python_rollingSum_cumsum(test_list, window= window)
    #     python_rollingSum_loop(test_list, window= window)
    #     python_rollingSum_loop_v2(test_list, window= window)
    
#%%     
    # Testing - testing validity of results
    
    ## Test data       
    test_list = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    
    ## Positive test case   
    
    # # window = 0
    # truthCase_0 = 
    
    # window = 1
    truth_list_1 = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    
    # window = 2
    truth_list_2 = [None, 3., 5., 7., 9., 11., 13., 15., 17., 19.]
    
    # window = 3
    truth_list_3 = [None, None, 6., 9., 12., 15., 18., 21., 24., 27.]
    
    # # window = 5
    # truthCase_5 = 
    
    # # window = 10
    # truthCase_10 = 
    
    truth_lists = [truth_list_1, truth_list_2, truth_list_3]
    #print(truth_lists)   
    
    
    for window in range(1, 4, 1): 
        print ('\nwindow:', window)
        
        # Grab truth case for window 
        truth_case = truth_lists[window-1]
        print(truth_case)
        
        # Run function for window 
        df_1 = pandas_df_rollingSum_loop(pd.DataFrame(test_list), window= window)
        df_2 = pandas_df_rollingSum_cumsum(pd.DataFrame(test_list), window= window)
        df_3 = pandas_df_rollingSum_rolling(pd.DataFrame(test_list), window= window)
        
        series_1 = pandas_series_rollingSum_loop(pd.Series(test_list), window= window)
        series_2 = pandas_series_rollingSum_cumsum(pd.Series(test_list), window= window)
        series_3 = pandas_series_rollingSum_rolling(pd.Series(test_list), window= window)
                
        numpy_1 = numpy_rollingSum_cumsum(np.array(test_list, dtype=float), window= window)
        numpy_2 = numpy_rollingSum_loop(np.array(test_list, dtype=float), window= window)
        numpy_3 = numpy_rollingSum_rolling(np.array(test_list, dtype=float), window= window)
                    
        python_1 = python_rollingSum_cumsum(test_list, window= window)
        python_2 = python_rollingSum_loop(test_list, window= window)
        python_3 = python_rollingSum_loop_v2(test_list, window= window)
        
        print(df_1)
        print(df_2)
        print(df_3)
        
        print(series_1)
        print(series_2)
        print(series_3)
        
        print(numpy_1)
        print(numpy_2)
        print(numpy_3)
        
        print(python_1)
        print(python_2)
        print(python_3)
      
        # Test cases for window 
        assert df_1.equals(df_2), "Test Failed"
        assert df_1.equals(df_3), "Test Failed"
        assert df_1.equals(pd.DataFrame(truth_case)), "Test Failed"
        
        assert series_1.equals(series_2), "Test Failed"
        assert series_1.equals(series_3), "Test Failed"
        assert series_1.equals(pd.Series(truth_case)), "Test Failed"
            
        assert np.array_equal(numpy_1, numpy_2, equal_nan=True), "Test Failed"
        assert np.array_equal(numpy_1, numpy_3, equal_nan=True), "Test Failed"
        assert np.array_equal(numpy_1, np.array(truth_case, dtype= float), equal_nan=True), "Test Failed"
        
        assert python_1 == python_2, "Test Failed" # Double check this is good to go and I don't have to iterate over items in the list 
        assert python_1 == python_3, "Test Failed"
        assert python_1 == truth_case, "Test Failed"


    ## Negative test case - Pending 

