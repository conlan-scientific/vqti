#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 19:35:15 2022

Load all tickers and generate signals dataframe to prepare data for simulation. 

@author: ellenyu

"""
from load_ellen import * 
from cci_ellen import *
from signal_ellen import *
from time import perf_counter

#%%
# Load the closing prices data 

df_prices = load_all_onecolumn()
#print(df)

#%%
# Load the transformed, indicator data 

# ## Method 1 - use pandas_df_cci function 
# df_ccimodified_pandas_df = pandas_df_ccimodified_rolling(df)
# #print(df_cci)


## Method 2  - use python_cci function 
# print('Method 2 - python cci applied to columns with for loop:\n')
# start = perf_counter()
# # Create a list of cci for each ticker
# list_cci=[]
# list_cci.append(df_prices.index)

# for column in df_prices.columns: 
#     cci = python_ccimodified_loop(df_prices[column].tolist()) # Because python_cci_loop processes one stock at a time
#     list_cci.append(cci)

# # print('verifying tranpose:', list_cci[1][:20], '\n')
# # print('tickers in list of ccis:', len(list_cci), '\n')
# # print('number of ccis per ticker:', len(list_cci[0]), '\n')

# df_ccimodified_python = pd.DataFrame(list_cci).T
# # print('verifying tranpose:', df_cci_python.iloc[:20,1], '\n')

# df_ccimodified_python.columns = df_prices.reset_index().columns
# # print('verifying column names:\n', df_cci_python.iloc[:, :3], '\n')
# # print('shape:', df_cci_python.shape, '\n')

# df_ccimodified_python = df_ccimodified_python.set_index('date')
# stop = perf_counter()
# print('elapsed time:', stop-start, 'seconds\n')
# # print('shape:', df_ccimodified_python.shape, '\n')
# print(df_ccimodified_python)


## Method 3 - create a load_all_onecolumn_with_cci function 
# print('Method 3 - python cci applied to columns with custom load function:\n')
# start = perf_counter()
# df_ccimodified = load_all_onecolumn_with_ccimodified()
# stop = perf_counter()
# print('elapsed time:', stop-start, 'seconds\n')
# print(df_ccimodified)


## Method 4 - use apply! [Chris suggestion]
print('Method 4 - python cci applied to columns with apply:\n')
start = perf_counter()
df_ccimodified = df_prices.apply(lambda col: python_ccimodified_loop(col.tolist()), axis=0)
stop = perf_counter()
print('elapsed time:', stop-start, 'seconds\n')
print(df_ccimodified)


## Observation 
# .apply method is fastest by 0.5 seconds. Can use .apply moving forward 

#%%
# Load the generated signals data

## Method 1 - Iterate over columns, and apply cci_signals_generator function
print('method 1 - iterate over columns:\n')

df_cci_signals = pd.DataFrame()
 
for column in df_ccimodified.columns: 
    cci_signals = cci_signals_generator_reversal(df_ccimodified[column]) # Because I wrote cci_signals_generator to process one stock at a time
    df_cci_signals = pd.concat([df_cci_signals, cci_signals], axis=1)

#print(df_cci_signals)

### Verify results 
#### Simple way 
for column in df_cci_signals[['AWU', 'AXC', 'BGN']].columns: 
    print(column, '\n', df_cci_signals[column].value_counts(), '\n')
    
#### Trickier way, but cleaner result 
df_melt = df_cci_signals.melt(var_name='columns', value_name='index')
df_tab = pd.crosstab(index=df_melt['index'], columns=df_melt['columns'])
print(df_tab.iloc[:, :3], '\n')


## Method 2 - Apply cci_signals_generator function to dataframe 
print('method 1 - apply to the dataframe:\n')

df_cci_signals = cci_signals_generator_reversal(df_ccimodified)

#print(df_cci_signals)

### Verify results 
#### Simple way 
for column in df_cci_signals[['AWU', 'AXC', 'BGN']].columns: 
    print(column, '\n', df_cci_signals[column].value_counts(), '\n')
    
#### Trickier way, but cleaner result 
df_melt = df_cci_signals.melt(var_name='columns', value_name='index')
df_tab = pd.crosstab(index=df_melt['index'], columns=df_melt['columns'])
print(df_tab.iloc[:, :3], '\n')

## Observation 
# Applying a function made for pd.Series on pd.DataFrame works. Use but keep tabs on it 

#%% 
# # Run through a simulator [largely Chris' implementaion]
# prices_df = df_prices
# signal_df = df_cci_signals

# assert prices_df.index.equals(signal_df.index), "Indices do not equal"

# max_assets = 20
# dt_index = prices_df.index
# cash = starting_cash = 100_000 #??? Underscores are acceptable in int?
# portfolio: List[str] = list()
# equity_curve = dict()


# # Pretend you are walking through time trading over the course of ten years
# for date in dt_index:

# 	signals = signal_df.loc[date] 

# 	stocks_im_going_to_buy: List[str] = signals[signals == 1].index.tolist() #??? Stock names are retained here?
# 	stocks_im_going_to_sell: List[str] = signals[signals == -1].index.tolist()
# 	stocks_im_holding: List[str] # This is determined by which stocks you bought previously [Pending, I think]

# 	# Mess with your cash and portfolio to sell stocks
# 	for stock in stocks_im_going_to_sell:
# 		cash += prices_df.loc[date, stock] * #number of shares of stock [Pending: where are we holding number of shares?]
# 		portfolio.pop(stock)
# 		stocks_im_holding.pop(stock)

# 	# What do I do with too many buy signals?
# 	# Mess with your cash and portfolio to buy stocks
# 	for stock in stocks_im_going_to_buy:
# 		# This is the "compound your gains and losses" approach to cash management
# 		# Also called the "fixed number of slots" approach
# 		cash_to_spend = cash / (max_assets - len(stocks_im_holding)) #[Pending: will run into DivisionByZeroError]

# 		# Straight from the book ... See page 68
# 		# self.free_position_slots = self.max_active_position - self.active_positions_count
# 		# cash_to_spend = self.cash / self.free_position_slots

# 		cash -= cash_to_spend
# 		portfolio.append(stock)
# 		stocks_im_holding.append(stock) 
#         #[Pending: are we doing to track number of shares of stock? ]

# 	equity_curve[date] = cash + # sum of my portfolio value [Pending]


# # Plot the equity curve [Pending]

# # Report the CAGR [Pending]
# calculate_cagr(equity_curve)

# # Measure the sharpe ratio [Pending]
# calculate_sharpe_ratio(equity_curve)

# # You're done.