#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:49:30 2022

First and foremost, simulate on more tickers.

Pending: 
    * trouble shoot nans in annualized returns and sharpe ratio
    * trouble shoot ValueError: Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.

What are ways to improve the current trading strategy? 
    * overlay macroeconomic metrics 
    * build better cash management strategy into simulator
    * better indicators - Note, chris argues there's no significant difference 
    between technical indicators 
    * more indicators and do some kind of voting method e.g. 
    ensemble methods, bayesian model averaging, or uniform weights 
    * learning algorithm 
    * ultimately, improve my simulator by 1) inputing better cash 
    management strategy, 2) getting a feel for what's importance to track 
    e.g. do we care about the number of bad trades (i.e. resulting in losses) 
    versus good trades, 3) building a simulator object and write tests

@author: ellenyu

"""

import pandas as pd
from cci_ellen import * 
from signal_ellen import * 
from simulate_ellen import *
from vqti.performance import calculate_cagr, calculate_sharpe_ratio
from pypm import metrics, signals, data_io, simulation

#%%

# Load in prices.csv 
## Over 12 million rows of data covering 6,305 stocks and date ranges from December 1997 to September 2021 
price_df = pd.read_csv('/Users/ellenyu/Desktop/UVA MSDS/Capstone/Coding/prices.csv', parse_dates = ['date'], index_col='date')

# Data wrangling 
## Drop all columns except ticker and price
price_df = price_df[['ticker', 'close_split_adjusted']]

## Before I assign, double check how many rows and columns I should expect to see 
# print(price_df.ticker.nunique()) #6305
# print(price_df.index.nunique()) #5969

## Pivot tickers values to column names 
price_df = price_df.pivot_table(index = 'date', columns='ticker', values='close_split_adjusted')
# print(price_df.shape) #(5969, 6305)

# Run technical indicator code on prices 
indicator_df = price_df.apply(lambda col: python_ccimodified_loop(col.tolist()), axis=0)

# # Calculate signals df (reversal strategy)
# signal_df = cci_signals_generator_reversal(indicator_df)

# Calculate signals df (momentum strategy)
signal_df = cci_signals_generator_momentum(indicator_df, upper_band = 2, lower_band = -1)

#%%
# Check signals df contains buy, sell, and hold signals 
melt_df = signal_df.melt(var_name='columns', value_name='index')
tab_df = pd.crosstab(index=melt_df['index'], columns=melt_df['columns'])
print(tab_df, '\n')

#%%
# Run my simualator 
cash_dict, portfolio_dict, equity_dict = quick_simulator(price_df = price_df, signal_df = signal_df)

# Visualize performance
plt.plot(cash_dict.keys(), cash_dict.values())
plt.title('Cash over Time')
plt.show()
plt.plot(portfolio_dict.keys(), portfolio_dict.values())
plt.title('Portfolio over Time')
plt.show()
plt.plot(equity_dict.keys(), equity_dict.values())
plt.title('Equity over Time')
plt.show()        

# Report cagr and sharpe - cci + simulator 
cagr = calculate_cagr(pd.Series(equity_dict))
print('MY annualized returns: {:.2%}'.format(cagr))

sharpe = calculate_sharpe_ratio(pd.Series(equity_dict))
print('MY sharpe ratio: {:.2f}'.format(sharpe))

# Report cagr and sharpe - spy benchmark 
spy = pd.read_csv('/Users/ellenyu/vqti/data/SPY.csv', parse_dates = ['date'], index_col ='date') #!!! read_csv has set_index and pd.to_datetime functionality 

cagr = calculate_cagr(spy.close)
print('SPY annualized returns: {:.2%}'.format(cagr))

sharpe = calculate_sharpe_ratio(spy.close)
print('SPY sharpe ratio: {:.2f}'.format(sharpe))

#%%
# Run chris's simulator
signal = signal_df
prices = price_df

preference = pd.DataFrame(0, index=prices.index, columns=prices.columns)


assert signal.index.equals(preference.index)
assert prices.columns.equals(signal.columns)
assert signal.columns.equals(preference.columns)

# Run the simulator
simulator = simulation.SimpleSimulator(
    initial_cash=100_000,
    max_active_positions=20,
    percent_slippage=0.0005,
    trade_fee=1,
)
simulator.simulate(prices, signal, preference)


# Print results
simulator.portfolio_history.print_position_summaries()
simulator.print_initial_parameters()
simulator.portfolio_history.print_summary()
simulator.portfolio_history.plot()
simulator.portfolio_history.plot_benchmark_comparison()