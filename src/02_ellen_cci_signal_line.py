#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:02:38 2022

@author: ellenyu

Pending: 
    * Visually inspect various signal lines 
    * Visualize the implemented signal line 
    * Raise an error if the input series is vastly different from normal distribution
    * Address the questions I've denoted below
    
"""
import pandas as pd
from vqti.load import load_eod
from cci_ellen import *
from load_ellen import *

#??? Tried to implement my thoughts into code as much as possible. I'm going to leave it be for now, but come back to my questions below at some point
def normal_distribution_signals_generator(series: pd.Series, num_std: int = 2) -> pd.Series:
    
    '''
    This function assume series has a normal distribution and generates buy/sell 
    signals when the value crosses over num_std. Using a num_std of 1 means high 
    risk (as 32% of values still lie beyond 1 standard deviation of the mean) and 
    using a num_std of 3 means low risk (2.5% of values lie beyond 3 standard 
    deviations of the mean). The default num_std is 2 (5% of values lie beyond 
    2 standard deviations of the mean). 

    '''
    # Calculate the standard deviation
    series_std = series.std() #??? Should this be a rolling std 
    series_mean = series.mean() # ??? Should this be a rolling mean
    print('series std:', series_std, '\n')
    print('series mean:', series_mean, '\n')  
    
    # Generate buy and sell signals
    print('{} std above mean:'.format(num_std), series_mean + (num_std * series_std), '\n')
    print('{} std below mean:'.format(num_std), series_mean - (num_std * series_std), '\n')
    print('series:\n', series, '\n')
    # E.g. buy if price is below 2 std of mean and sell if price is above 2 std of mean
    signals = -1 * (series > series_mean+(num_std * series_std)) \
                + 1 * (series < series_mean -(num_std * series_std)) #??? Not sure why this line of code works, but I've checked across 3 tickers that this works 
    print(signals.value_counts())
    print('signals:', signals, '\n')

    assert type(signals) == pd.Series, "Output array is not same type as input array"
    assert len(signals) == len(series), "Output array is not same length as input array"
    
    return signals

def cci_signals_generator(series: pd.Series) -> pd.Series:
    
    '''
    cci is like a z-score which means 1 contains roughly 68% of the data, 2 contains 
    roughly 95% of the data, 3 contains roughly 97% of the data, and 4 contains roughly 
    99% of the data. 
    
    If cci > 3 then buy, if cci < -2 sell [Chris suggestion] # ??? Double check that this is not the other way around
    '''
    # Generate buy and sell signals
    print('series:\n', series, '\n')
    print('where series > 3:\n', series[series>3], '\n')
    print('where series <-2:\n', series[series<-2], '\n')
    #Sell when price is greater than 3 z-scores and buy when price is less than -2 z-scores
    signals = -1 * (series > 3) \
                + 1 * (series <-2) #??? Not sure why this line of code works, but I've checked across 3 tickers that this works 
    print(signals.value_counts())
    print('signals:', signals, '\n')
    
    assert type(signals) == pd.Series, "Output array is not same type as input array"
    assert len(signals) == len(series), "Output array is not same length as input array"
    
    return signals
    
if __name__ == '__main__':

    # Testing signals generator functions 
    df = load_eod('AXC')
    #print(df.head())
    
    # For any series with a normal distribution 
    normal_distribution_signals_generator(df.close, num_std=1)
    
    # For cci which is already a z-score
    ## Turn df into cci 
    cci = python_ccimodified_loop(df.close.tolist(), window=5)         
    ## Run signals generator function on cci 
    cci_signals_generator(pd.Series(cci))
    
#%%    
    ## Testing whether to use cci or cci modified 
    #cci = python_cci_loop(df.close.tolist())
    #cci_modified = python_ccimodified_loop(df.close.tolist())

    ### Check out summary statistics 
    #print('cci Lambert:\n', pd.Series(cci).describe(), '\n')
    #print('cci modified:\n', pd.Series(cci_modified).describe(), '\n')
    #print('testing:\n', pd.Series(cci_modified).describe()*(1/0.015), '\n')
    
    #### Between original and modified versions of indicator
    ## AXC
    # cci original:
    #   count    2497.000000
    # mean       19.023824
    # std       108.141255
    # min      -294.205210
    # 25%       -63.003968
    # 50%        31.069328
    # 75%       101.521979
    # max       426.438517
    # dtype: float64 
    
    # cci modified:
    #  count    2497.000000
    # mean        0.285357
    # std         1.622119
    # min        -4.413078
    # 25%        -0.945060
    # 50%         0.466040
    # 75%         1.522830
    # max         6.396578
    # dtype: float64 
    
    # testing:
    #  count    166466.666667
    # mean         19.023824
    # std         108.141255
    # min        -294.205210
    # 25%         -63.003968
    # 50%          31.069328
    # 75%         101.521979
    # max         426.438517
    # dtype: float64 
    
    #### Between tickers
    ## AWU
    # cci modified:
    #  count    2497.000000
    # mean        0.110201 # mean 0.11
    # std         1.719496 # std 1.7
    # min        -8.960236 # range_min = -8.96
    # 25%        -1.187561
    # 50%         0.185836
    # 75%         1.404015
    # max         7.771026
    # dtype: float64 
    
    ## AXC 
    # cci modified:
    #  count    2497.000000
    # mean        0.285357 # mean 0.28 
    # std         1.622119 # std 1.62
    # min        -4.413078 # range_min -4.41
    # 25%        -0.945060
    # 50%         0.466040
    # 75%         1.522830
    # max         6.396578
    # dtype: float64 
    
    ## BGN 
    # cci modified:
    #  count    1521.000000
    # mean        0.401470 # mean 0.401
    # std         1.672305 # std 1.67
    # min        -5.771888 # range_min -5.77
    # 25%        -0.764932
    # 50%         0.734211
    # 75%         1.615385
    # max         5.892359
    # dtype: float64 
    
    ### Based on what I know about z-scores, s-scores have mean 0 and std 1 and are unbounded 
    #### First, built a load all function - let's check out what it does
    df_all = load_all()
    #print(df_all.shape)
    #print(df_all.head(21))

    #### Run technical indicator by ticker - currently using work around function 
    df_all_cci = load_all_with_cci()
    df_all_ccimodified = load_all_with_ccimodified()
    # print(df_all_cci.shape)
    # print(df_all_cci.head(21))
    # print(df_all_ccimodified.shape)
    # print(df_all_ccimodified.head(21))
    
    #### Checking out the mean, std, and range across the tickers 
    summary = df_all_cci[['ticker', 'cci']].groupby('ticker').agg(['mean', 'std', 'min', 'max'])  
    #print(summary)
    # print(type(summary))
    # print(summary.columns)
    
    summary_modified = df_all_ccimodified[['ticker', 'cci']].groupby('ticker').agg(['mean', 'std', 'min', 'max'])  
    # print(summary_modified)
    # print(type(summary_modified))
    # print(summary_modified.columns)
    
    #### Checking out the average mean and average std across the tickers  
    summary.columns = summary.columns.droplevel()
    print(summary[['mean', 'std']].describe(), '\n')
    
    summary_modified.columns = summary_modified.columns.droplevel()
    print(summary_modified[['mean', 'std']].describe(), '\n')
    
    #### Observations 
    #Across the tickers, the original cci has an average std of 108 with a std of 2 so, use the modified cci 
    #Across the tickers, the modified cci has an average mean of 0.257 with a std of .121
    #Across the tickers, the modified cci has an average std of 1.632 with a std of .030 so, a little bigger than the z-score std of 1 
