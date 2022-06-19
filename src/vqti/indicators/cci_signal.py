#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:39:28 2022

cci signals generator functions for sharing 

@author: ellenyu
"""

import pandas as pd
from vqti.indicators.cci import *

def create_cci_signal_pandas (series: pd.Series, strategy_type: str= 'reversal', upper_band: int=3, lower_band: int=-3, window: int=20, factor: int=1) -> pd.Series: 
    '''
    this functions calls cci generator function and generates signals 
    '''
    cci = pandas_cci_rolling(series, window = window, factor = factor)
    
    if strategy_type == 'reversal':
        signals = -1 * (cci > upper_band) \
                    + 1 * (cci < lower_band)
    elif strategy_type == 'momentum':
        signals = 1 * (cci > upper_band) \
                    - 1 * (cci < lower_band)
                
    #assert type(signals) == pd.Series, "Output array is not same type as input array"
    assert len(signals) == len(series), "Output array is not same length as input array"
    
    return signals

def create_cci_signal_python(series: pd.Series, strategy_type: str= 'reversal', upper_band: int=3, lower_band: int=-3, window: int=20, factor: int=1) -> pd.Series: 
    '''
    this functions calls cci generator function and generates signals 
    '''
    cci = pd.Series(python_cci_loop(series.tolist(), window = window, factor = factor)) # Is it a cop out to turn this immediately to a pd.Series? Return to this
    
    if strategy_type == 'reversal':
        signals = -1 * (cci > upper_band) \
                    + 1 * (cci < lower_band)
    elif strategy_type == 'momentum':
        signals = 1 * (cci > upper_band) \
                    - 1 * (cci < lower_band)
                
    #assert type(signals) == pd.Series, "Output array is not same type as input array"
    assert len(signals) == len(series), "Output array is not same length as input array"
    
    return signals

def create_cci_signal_numpy(series: pd.Series, strategy_type: str= 'reversal', upper_band: int=3, lower_band: int=-3, window: int=20, factor: int=1) -> pd.Series: 
    '''
    this functions calls cci generator function and generates signals 
    '''
    cci = pd.Series(numpy_cci_rolling(series.to_numpy(), window = window, factor = factor)) # Is it a cop out to turn this immediately to a pd.Series? Return to this
    
    if strategy_type == 'reversal':
        signals = -1 * (cci > upper_band) \
                    + 1 * (cci < lower_band)
    elif strategy_type == 'momentum':
        signals = 1 * (cci > upper_band) \
                    - 1 * (cci < lower_band)
                
    #assert type(signals) == pd.Series, "Output array is not same type as input array"
    assert len(signals) == len(series), "Output array is not same length as input array"
    
    return signals

if __name__ == '__main__':
    
    url = 'https://s3.us-west-2.amazonaws.com/public.box.conlan.io/e67682d4-6e66-48f8-800c-467d2683582c/0b40958a-fa6f-448f-acbf-9d5478308cf5/prices.csv'
    
    price_df = pd.read_csv(url, parse_dates = ['date'], index_col ='date')
    
    price_df = price_df[['ticker', 'close_split_adjusted']]

    price_df = price_df.pivot_table(index = 'date', columns='ticker', values='close_split_adjusted')
    
    signals_ticker = create_cci_signal_pandas(price_df.ASYS)
    
    plt.plot(price_df.ASYS )
    plt.plot(signals_ticker *10)
    plt.show()
    
    # There is a line in the simulator that prvents us from buying something we already own so, e.g. repeated
    # buy signals is not a problem but, the visual is a beast. Try to implement a signals function that alternates
    # between buy and sells signals 

