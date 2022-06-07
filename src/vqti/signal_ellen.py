#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:04:24 2022

[5-31-22] Created cci_signals_generator that takes strategy type as a parameter. 

@author: ellenyu
"""

import pandas as pd

def cci_signals_generator_momentum(series: pd.Series, upper_band: int = 3, lower_band: int = -2) -> pd.Series:
    
    '''
    cci is like a z-score which means 1 contains roughly 68% of the data, 2 contains 
    roughly 95% of the data, 3 contains roughly 97% of the data, and 4 contains roughly 
    99% of the data. 
    
    If cci > 3 then buy, if cci < -2 sell [Chris suggestion] 
    '''
    # Generate buy and sell signals
    # print('series:\n', series, '\n')
    # print('where series > 3:\n', series[series>3], '\n')
    # print('where series <-2:\n', series[series<-2], '\n')
    #Buy when price is greater than 3 z-scores and sell when price is less than -2 z-scores
    signals = 1 * (series > upper_band) \
                - 1 * (series < lower_band) #??? Not sure why this line of code works, but I've checked across 3 tickers that this works 
    # print(signals.value_counts())
    # print('signals:', signals, '\n')
    
    #assert type(signals) == pd.Series, "Output array is not same type as input array"
    assert len(signals) == len(series), "Output array is not same length as input array"
    
    return signals

def cci_signals_generator_reversal(series: pd.Series, upper_band: int = 3, lower_band: int = -2) -> pd.Series:
    
    '''
    cci is like a z-score which means 1 contains roughly 68% of the data, 2 contains 
    roughly 95% of the data, 3 contains roughly 97% of the data, and 4 contains roughly 
    99% of the data. 
    
    If cci > 3 then sell, if cci < -2 buy
    '''
    # Generate buy and sell signals
    # print('series:\n', series, '\n')
    # print('where series > 3:\n', series[series>3], '\n')
    # print('where series <-2:\n', series[series<-2], '\n')
    #Sell when price is greater than 3 z-scores and buy when price is less than -2 z-scores
    signals = -1 * (series > upper_band) \
                + 1 * (series < lower_band) #??? Not sure why this line of code works, but I've checked across 3 tickers that this works 
    # print(signals.value_counts())
    # print('signals:', signals, '\n')
    
    # assert type(signals) == pd.Series, "Output array is not same type as input array"
    assert len(signals) == len(series), "Output array is not same length as input array"
    
    return signals

def cci_signals_generator(series: pd.Series, strategy_type: str, upper_band: int, lower_band: int) -> pd.Series:
    
    '''
    cci is like a z-score which means 1 contains roughly 68% of the data, 2 contains 
    roughly 95% of the data, 3 contains roughly 97% of the data, and 4 contains roughly 
    99% of the data. 
    '''
    if strategy_type == 'reversal':
        signals = -1 * (series > upper_band) \
                    + 1 * (series < lower_band)
    elif strategy_type == 'momentum':
        signals = 1 * (series > upper_band) \
                    - 1 * (series < lower_band)
                
    #assert type(signals) == pd.Series, "Output array is not same type as input array"
    assert len(signals) == len(series), "Output array is not same length as input array"
    
    return signals