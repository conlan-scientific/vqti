#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 08:54:17 2022

Explored the performance of cci via two simulators. 

Note, on 100 tickers, it takes ~0.14 seconds to run quick simulator and 
~3 seconds to run chris simulator. 

Pending: 
    * double check if quick simulator is incorrect because the results 
    (equity curve, cagr, sharpe ratio) are horrible aka write better simulator code   

@author: ellenyu

"""
import pandas as pd
import numpy as np
from load_ellen import * 
from cci_ellen import * 
from signal_ellen import *
from simulate_ellen import * 
from vqti.performance import calculate_cagr, calculate_sharpe_ratio
from pypm import metrics, signals, data_io, simulation

# Load prices df 
price_df = load_all_onecolumn()
#print(price_df, '\n')

# Calculate technical indicator, cci (window defaults to 20)
indicator_df = price_df.apply(lambda col: python_ccimodified_loop(col.tolist()), axis=0)
#print(indicator_df, '\n')

# # Calculate signals df (reversal strategy)
# signal_df = cci_signals_generator_reversal(indicator_df)
# #print(signal_df, '\n')

# Calculate signals df (momentum strategy)
signal_df = cci_signals_generator_momentum(indicator_df, upper_band = 2, lower_band = -1)
#print(signal_df, '\n')

# Check signals df contains buy, sell, and hold signals 
melt_df = signal_df.melt(var_name='columns', value_name='index')
tab_df = pd.crosstab(index=melt_df['index'], columns=melt_df['columns'])
#print(tab_df, '\n')

#%%
# Method - Function
cash_dict, portfolio_dict, equity_dict = quick_simulator(price_df = price_df, signal_df = signal_df) # Completed in 8785.879 milliseconds = ~0.14 minutes 

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
# Method - Class 
preference_df = pd.DataFrame(0, index=price_df.index, columns=price_df.columns)

# Do nothing on the last day
signal_df.iloc[-1] = 0

assert signal_df.index.equals(preference_df.index)
assert price_df.columns.equals(signal_df.columns)
assert signal_df.columns.equals(preference_df.columns)


# Run the simulator

simulator = simulation.SimpleSimulator(
    initial_cash=100_000,
    max_active_positions=20,
    percent_slippage=0.0005,
    trade_fee=1,
)
simulator.simulate(price = price_df, signal = signal_df, preference  = preference_df) #Completed simulate in 2705.678 milliseconds

# when it works, these are the inputs for jensens alpha function 
# return series input: 2010-01-04         NaN
# 2010-01-05    0.000000
# 2010-01-06    0.000000
# 2010-01-07    0.000000
# 2010-01-08    0.000000
  
# 2019-12-24   -0.000199
# 2019-12-26    0.000404
# 2019-12-27   -0.000030
# 2019-12-30   -0.005512
# 2019-12-31    0.001714
# Length: 2516, dtype: float64 

# benchmark input: date
# 2010-01-05         NaN
# 2010-01-06    0.000704
# 2010-01-07    0.004212
# 2010-01-08    0.003322
# 2010-01-11    0.001396
  
# 2019-12-24    0.000031
# 2019-12-26    0.005309
# 2019-12-27   -0.000248
# 2019-12-30   -0.005528
# 2019-12-31    0.002426
# Name: close, Length: 2515, dtype: float64 

# df:                    0     close
# 2010-01-06  0.000000  0.000704
# 2010-01-07  0.000000  0.004212
# 2010-01-08  0.000000  0.003322
# 2010-01-11  0.000000  0.001396
# 2010-01-12  0.000000 -0.009370
#              ...       ...
# 2019-12-24 -0.000199  0.000031
# 2019-12-26  0.000404  0.005309
# 2019-12-27 -0.000030 -0.000248
# 2019-12-30 -0.005512 -0.005528
# 2019-12-31  0.001714  0.002426

# [2514 rows x 2 columns] 

# clean_returns: 2010-01-06    0.000000
# 2010-01-07    0.000000
# 2010-01-08    0.000000
# 2010-01-11    0.000000
# 2010-01-12    0.000000
  
# 2019-12-24   -0.000199
# 2019-12-26    0.000404
# 2019-12-27   -0.000030
# 2019-12-30   -0.005512
# 2019-12-31    0.001714
# Name: 0, Length: 2514, dtype: float64 

# clean_benchmarks:                close
# 2010-01-06  0.000704
# 2010-01-07  0.004212
# 2010-01-08  0.003322
# 2010-01-11  0.001396
# 2010-01-12 -0.009370
#              ...
# 2019-12-24  0.000031
# 2019-12-26  0.005309
# 2019-12-27 -0.000248
# 2019-12-30 -0.005528
# 2019-12-31  0.002426

#Print results
# simulator.portfolio_history.print_position_summaries()
simulator.print_initial_parameters()
simulator.portfolio_history.print_summary()
simulator.portfolio_history.plot()
simulator.portfolio_history.plot_benchmark_comparison()

