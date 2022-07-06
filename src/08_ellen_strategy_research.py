#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:37:28 2022

Does near term change in volume of the stock have an impact on the return?

to dos: 
    - try this with regression 
    - standardize instead of min-max scale 

@author: ellenyu
"""
import pandas as pd 
import seaborn as sns

# load data 
AWU = pd.read_csv("/Users/ellenyu/vqti/data/eod/AWU.csv", index_col= 'date', parse_dates=['date'])

# generate features 
def generate_volume_features(series: pd.Series, n:int = 90) -> pd.Series: 
    '''
    Takes a series of date-index volume metrics and returns a series of date-indexed
    90-day rolling standard deviations. This is applying min max scaling and a rolling
    std function. 
    
    '''
    scaled = (series-series.min())/(series.max()-series.min())
    
    return scaled.rolling(n).std()

#generate X data frame 
X = generate_volume_features(AWU.volume)

# generate event indices 
## enter at 7, 14, 30, 60, 90 etc. day low 
def get_nday_low (series:pd.Series, n:int = 14) -> pd.Series: 
    '''
    Returns data and price if price is nday low. 
    '''
    # assuming pd.Series subtraction will be faster than iterating through rows 
    df = pd.concat([series, series.rolling(n).min()], axis=1)

    # if difference between prices and rollingmin prices is 0
    df['diff'] = df.iloc[:, 0] - df.iloc[:, 1]
    
    # return the prices and the dates 
    return df.query('diff ==0').iloc[:, 0]   

# apply to X data frame 
event_index = get_nday_low(AWU.close).index
X = X[event_index]

# generate labels
def generate_labels(price_series: pd.Series, event_series: pd.Series, upper_x: int=1.2, lower_x: int=0.8, vertical_barrier: int=90) -> pd.Series:
    '''
    Takes a series of dates and prices and if 
    1) 1.2x price, generate 1 
    2) 0.8x price, generate -1
    3) does not meet any of the above in 90 days, generate a 0
    '''
    
    labels_dict = {}
            
    for date, price in event_series.iteritems():
        #print(date)
        #print(price)
        forward_index = date + pd.Timedelta(days = vertical_barrier)
        #print(forward_index, '\n')
        
        forward_series = price_series[date:forward_index]
        #print(forward_series, '\n')

        upper_barrier = upper_x * price 
        lower_barrier = lower_x * price 
        
        barrier_breached = forward_series[(forward_series>upper_barrier) | (forward_series<lower_barrier)]
        #print(barrier_breached)
        
        if len(barrier_breached) == 0: 
            labels_dict[date] = 0
        else: 
            if barrier_breached[0] > price:
                labels_dict[date] = 1 
            else:
                labels_dict[date] = -1
    
    return pd.Series(labels_dict, name = 'labels')

event_labels = generate_labels(price_series = AWU.close, event_series = get_nday_low(AWU.close))

# join on index 
research_df = pd.concat([X, event_labels], axis=1)

# plot 
sns.scatterplot(x = 'volume', y = 'labels', data = research_df)

