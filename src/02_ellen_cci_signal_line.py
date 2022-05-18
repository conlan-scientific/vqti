#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:02:38 2022

@author: ellenyu

To do: 
    - put rolling function into cci - DONE 
    - technically, I should keep testing cci 
    - write a signal converter function 
        - tehcnically, I should visualize and analyze the results  
"""
import pandas as pd
from vqti.load import load_eod

def signals_generator (series: pd.Series, std: int = 2) -> pd.Series:
    '''

    Parameters
    ----------
    series : pd.Series
        this function takes a moving metric series.
    std : int, optional
        std is loosely bounded between 1 and 4. A std of 1 means high risk. A std of 3 means low risk. The default is 2.

    Returns
    -------
    a pd.Series of buy and sell signals denoted as '1' and '0'.

    '''
    
    # Calculate the standard deviation
    series_std = series.std()
    print('series std:', series_std, '\n')
    
    # Generate buy and sell signals
    signals = 1 * (series > std * series_std) - 1 * (std * series_std < 0)
    print('signals:', signals, '\n')
    print(signals.value_counts())
    
    assert type(signals) == pd.Series, "Output array is not same type as input array"
    assert len(signals) == len(series), "Output array is not same length as input array"
    
if __name__ == '__main__':
    
    df = load_eod('AWU')
    #print(df.head())
    
    signals_generator(df.close, std=4)
