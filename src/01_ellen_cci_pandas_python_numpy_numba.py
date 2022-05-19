#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:13:55 2022

@author: ellenyu

To do's:
    * Troubleshoot numba implementation 
    * Realized in pandas implementations, mad is a rolling number. Incorporate in other functions as well - Complete for key functions of 5/18/22
    * Pass unit tests of quality between methods and against truth case - Complete for key functions as of 5/18/22 
    * Estimate time complexity
    * Is there a best practices in terms of float precision?
"""
from vqti.load import load_eod 
from vqti.profile import time_this 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from typing import List
from numba import jit

@time_this 
def pandas_df_cci(df: pd.DataFrame, window: int=20) -> pd.DataFrame: 
	
    # Find the moving average of closing price 
    df['sma'] = df['close'].rolling(window).mean()
    #print(df.head(21), '\n')
	
    # Calculate the mean absolute deviation
	#df['mad'] = df['close'].rolling(window).apply(lambda x: pd.Series(x).mad()) #??? According to documentation .mad() takes a series, but the following works too
    df['mad'] = df['close'].rolling(window).apply(lambda x: x.mad()) 
    #print(df.head(21), '\n')

    # Put it all together 

    # 1st order calculation expressed in MAD Z-score units
    # TODO: Remove lambert's constant everywhere for easy interpretation
    df['cci'] = (df['close'] - df['sma']) / (0.015 * df['mad']) # 0.015 is Lambert's constant 
    #print(df.head(21), '\n')
    
    return df 


@time_this
def pandas_series_cci(close: pd.Series, window: int=20) -> pd.Series: 
	
    # Find the moving average of closing price 
    sma: pd.Series = close.rolling(window).mean()
    #print(type(sma), '\n')
    assert type(sma) == pd.Series, "Does not equal"
    #print(sma.head(21),'\n')
    
    # Calculate the mean absolute deviation
    mad: pd.Series = close.rolling(window).apply(lambda x: pd.Series(x).mad())
    #print(type(mad), '\n')
    assert type(mad) == pd.Series, "Does not equal"
    #print(mad.head(21),'\n')

	# Put it all together
    cci: pd.Series = (close-sma) / (0.015 * mad)
    #print(type(cci), '\n')
    assert type(cci) == pd.Series, "Does not equal"
    #print(cci.head(21),'\n')

    return cci


@time_this
def python_cci_listcomp (close: List[float], window: int=20) -> List[float]:
    
    # Find the moving average of closing price 
    ##??? Assuming there is no cumulative sum function in pure python...
    sma = [0] * (window-1)
    #print(sma, '\n')
    accum = sum(close[:window])
    sma.append(accum / window)
    #print(sma, '\n')
    for i in range(window, len(close)):
        accum += close[i] # Add one index
        accum -= close[i-window] # Subtract one index so that it is always window number of indices
        sma.append(accum / window)
    #print(sma[:window+1],'\n')
     
    assert len(sma) == len(close), "Does not equal"
    
    
    # Calculate the mean absolute deviation
    #print(close[:window+1])
    mean = sum(close) / len(close)
    #print('mean:', mean, '\n')
    deviations = [x - mean for x in close]
    #print(deviations[:window+1])
    abs_deviations = [abs(x) for x in deviations]
    #print(abs_deviations[:window+1])
    mad = sum(abs_deviations)/ len(abs_deviations)
    #print('mad:', mad)
    
    
    # Put it all together
    ## Subtract two lists 
    leading = [x1 - x2 for (x1, x2) in zip(close, sma)]
    #print('close:', close[window-2:window+1], '\n')
    #print('sma:', sma[window-2:window+1], '\n')
    #print('lead:', leading[window-2:window+1], '\n')
    cci = [x/(0.015 * mad) for x in leading]
    #print('mad:', 0.015 * mad, '\n')
    #print('cci:', cci[window-2:window+1], '\n')

    
    return cci


@time_this
def python_cci_map (close: List[float], window: int=20) -> List[float]:
   
    # Find the moving average of closing price 
    ##??? Assuming there is no cumulative sum function in pure python...
    sma = [0] * (window-1)
    #print(sma, '\n')
    accum = sum(close[:window])
    sma.append(accum / window)
    #print(sma, '\n')
    for i in range(window, len(close)):
        accum += close[i] # Add one index
        accum -= close[i-window] # Subtract one index so that it is always window number of indices
        sma.append(accum / window)
    #print(sma[:window+1],'\n')
    
    assert len(sma) == len(close), "Does not equal" 


    # Calculate the mean absolute deviation
    mean = sum(close) / len(close)
    #print('mean:', mean, '\n')
    deviations = list(map(lambda x: x-mean, close))
    #print(deviations[:window+1], '\n')
    abs_deviations = list(map(lambda x: abs(x), deviations))
    #print(abs_deviations[:window+1], '\n')
    mad = sum(abs_deviations)/ len(abs_deviations)
    #print('mad:', mad)
    
    
    # Put it all together
    ## Subtract two lists 
    leading = list(map(lambda x,y: x-y, close, sma))
    #print('close:', close[window-2:window+1], '\n')
    #print('sma:', sma[window-2:window+1], '\n')
    #print('lead:', leading[window-2:window+1], '\n')
    cci = list(map(lambda x: x/(0.015*mad), leading))
    #print('mad:', 0.015 * mad, '\n')
    #print('cci:', cci[window-2:window+1], '\n')
    
    
    return cci 


@time_this
def numpy_cci (close: np.ndarray, window: int=20) -> np.ndarray:
    
    # Find the moving average of closing price
    accum = close.cumsum()
    delta_accum = accum[window:] - accum[:-window]
    sma = delta_accum / window
    sma = np.pad(sma, (window,0), 'constant')
    #print('sma:', sma[:window+1], '\n')
    
    assert len(sma) == len(close), 'Does not equal'
   
    
    # Calculate the mean absolute deviation
    mean: int = np.mean(close)
    #print('mean:', mean, '\n')
    deviations: List = close - mean
    #print('deviations:', deviations, '\n')
    abs_deviations: List = np.absolute(deviations)
    #print('absolute deviations:', abs_deviations, '\n')
    mad: int = np.mean(abs_deviations)
    #print('mad:', mad, '\n')

    
    # Put it all together
    cci = (close - sma)/(0.015 * mad)
    #print('cci:', cci[:window+1], '\n')
    
    return cci

    
@jit(nopython=True)
def _numba_cci (close: np.ndarray, window: int=20) -> np.ndarray:
    # Find the moving average of closing price
    accum = close.cumsum()
    delta_accum = accum[window:] - accum[:-window]
    sma = delta_accum / window
    # TODO: try removing the pad
    sma = np.pad(sma, (window,0), 'constant')
    #print('sma:', sma[:window+1], '\n')
    
    # TODO: Don't even think about defensive programming inside numba
    assert len(sma) == len(close), 'Does not equal'
   
    
    # Calculate the mean absolute deviation
    mean: int = np.mean(close)
    #print('mean:', mean, '\n')
    deviations: List = close - mean
    #print('deviations:', deviations, '\n')
    abs_deviations: List = np.absolute(deviations)
    #print('absolute deviations:', abs_deviations, '\n')
    mad: int = np.mean(abs_deviations)
    #print('mad:', mad, '\n')

    
    # Put it all together
    cci = (close - sma)/(0.015 * mad)
    #print('cci:', cci[:window+1], '\n')
    
    return cci

# # Do a once-over to get the compilation
# _numba_cci(np.array([1,2,3,4,5,6,7,8,9,10]), window=3)


# @time_this
# def numba_cci (close: np.ndarray, window: int=20) -> np.ndarray:
#     return _numba_cci (close, window=window)
    
    
if __name__ == '__main__':
    
	   
    # Load data
    df = load_eod('AWU')

    pandas_df = pandas_df_cci(df)
    pandas_series = pandas_series_cci(df.close)
    numpy = numpy_cci(df.close.to_numpy())
    python_listcomp = python_cci_listcomp(df.close.tolist())
    python_map = python_cci_map(df.close.tolist())
    
    #Tests
    # Pandas methods equal each other
    assert len(pandas_df.cci) == len (pandas_series), "Does not equal"
    for i in range(len(pandas_df.cci)):
        #print(i)
        np.testing.assert_equal(pandas_df.cci[i], pandas_series[i]), "Does not equal"
        
    # Python methods equal each other
    assert len(python_listcomp) == len (python_map), "Does not equal"
    for i in range(len(python_listcomp)):
        #print(i)
        np.testing.assert_equal(python_listcomp[i], python_map[i]), "Does not equal"
    
    # Numpy and numba methods equal each other - Pending 
    # Pandas and Python and Numpy methods equal each other - Pending
    assert len(python_listcomp) == len (numpy) == len (numpy), "Do not equal"
    for i in range(len(python_listcomp)):
        print(i)
        np.testing.assert_equal(round(python_listcomp[i],8), round(numpy[i],8)), "Does not equal"



        