#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:44:34 2022

Looking at volume distributions 

@author: ellenyu
"""

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

# load the data 
url = 'https://s3.us-west-2.amazonaws.com/public.box.conlan.io/e67682d4-6e66-48f8-800c-467d2683582c/0b40958a-fa6f-448f-acbf-9d5478308cf5/prices.csv'
df = pd.read_csv(url, index_col = 'date', parse_dates=['date'])

# reduce to columns 
volume_df = df[['ticker', 'volume']].reset_index()

# change the way the data is structured 
volume_df = volume_df.pivot(index ='date', columns ='ticker', values = 'volume')

# double-check nans 
## No nans in the data file 
df.isnull().sum()

## however, some tickrs have 25 years of data whereas other tickrs have 1% of that 
percent_dates = volume_df.isnull().sum()/volume_df.shape[0]*100
percent_dates.sort_values(ascending=False)

## double-check what data I have in case I want to start dropping data 
fig, ax  = plt.subplots(2)
sns.boxplot(x = percent_dates, ax=ax[0])
sns.violinplot(x = percent_dates, bw=0.01, ax=ax[1])
plt.xlabel('percentage of dates')
#sns.violinplot(x = percent_dates, cut=0, ax=ax[1])
## Takeaways - half of the tickers have 20 years of data or more. 75% of the data have 13 years of data or more. 
## Note, the default violin plot uses smoothing / density estimation, which might mislead. 
## Use a narrow bandwidth to reduce smoothing. The cut parameters cuts the density estimation to the range of actual values. 

# visualize density of volume 
for i in volume_df.columns[:5]: 
    sns.histplot(x = volume_df[i])
    plt.show()

# visualize volume over time
for i in volume_df.columns[:5]: 
    sns.lineplot(x = volume_df[i])
    plt.show()



