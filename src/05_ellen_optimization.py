#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 08:09:03 2022

To do:
    1) Do grid search 
    2) Get a df and start to analyze it 
    
    What do I want to test? window, reversal vs. momentum strategy, where to cut off bands 
    
    What do I need to do? Build a funtion that takes parameters, feed into it the necessary steps to do simulation, run simulation 
    
Encountered: 
    * AssertionError: Cannot buy zero or negative shares. - I think I circumvented this by commenting out the line of code. Return to this. 
    
    * Simulating ... 1 momentum 1 4 20 - I think I ignored this warning and it returned nans for sharpe ratio. Return to this. 
    /Users/ellenyu/vqti/src/pypm/metrics.py:78: RuntimeWarning: invalid value encountered in double_scalars
    return (cagr - benchmark_rate) / volatility

    * assert self.cash >= 0, 'Spent cash you do not have.' - I think I circumvented this by raising initial_cash. Return to this. 
    AssertionError: Spent cash you do not have.
    
@author: ellenyu

"""


import pandas as pd
from load_ellen import * 
from cci_ellen import * 
from signal_ellen import * 
from pypm import metrics, signals, data_io, simulation
from typing import List, Dict, Any
from time import perf_counter

# Load in prices_df - 100 tickers
price_df = load_all_onecolumn()
#preference = pd.DataFrame(0, index=price_df.index, columns=price_df.columns)
preference = price_df.apply(metrics.calculate_rolling_sharpe_ratio, axis=0)

#%%
# # Load in prices_df - extended tickers
# price_df = pd.read_csv('/Users/ellenyu/Desktop/UVA MSDS/Capstone/Coding/prices.csv', parse_dates = ['date'], index_col='date')
# price_df = price_df[['ticker', 'close_split_adjusted']]
# price_df = price_df.pivot_table(index = 'date', columns='ticker', values='close_split_adjusted')
# #preference = pd.DataFrame(0, index=prices.index, columns=prices.columns)
# preference = price_df.apply(metrics.calculate_rolling_sharpe_ratio, axis=0)

#%%
## Following Chris' model 
def run_simulation (window: int, strategy_type: str, lower_band: int, upper_band: int, max_active_positions: int) -> Dict[str, Any]: 
    
    # Generate signals
    indicator_df = price_df.apply(lambda col: python_ccimodified_loop(col.tolist(), window=window), axis=0)
    signal_df = cci_signals_generator(indicator_df, strategy_type, upper_band, lower_band)
    
    # Do nothing on the last day
    signal_df.iloc[-1] = 0

    # assert signal.index.equals(preference.index)
    # assert prices.columns.equals(signal.columns)
    # assert signal.columns.equals(preference.columns)

    # Run the simulator
    simulator = simulation.SimpleSimulator(
        initial_cash=100_000,
        max_active_positions=max_active_positions,
        percent_slippage=0.0005,
        trade_fee=1,
    )
    simulator.simulate(price = price_df, signal = signal_df, preference = preference)


    portfolio_history = simulator.portfolio_history
    return {
        # Performance metrics
        'percent_return': portfolio_history.percent_return,
        'spy_percent_return': portfolio_history.spy_percent_return,
        'cagr': portfolio_history.cagr,
        'volatility': portfolio_history.volatility,
        'sharpe_ratio': portfolio_history.sharpe_ratio,
        'spy_cagr': portfolio_history.spy_cagr,
        'excess_cagr': portfolio_history.excess_cagr,
        'jensens_alpha': portfolio_history.jensens_alpha,
        'dollar_max_drawdown': portfolio_history.dollar_max_drawdown,
        'percent_max_drawdown': portfolio_history.percent_max_drawdown,
        'log_max_drawdown_ratio': portfolio_history.log_max_drawdown_ratio,
        'number_of_trades': portfolio_history.number_of_trades,
        'average_active_trades': portfolio_history.average_active_trades,
        'final_equity': portfolio_history.final_equity,
        
        # Grid search parameters
        'window_length': window,
        'strategy_type': strategy_type,
        'lower_band': lower_band, 
        'upper_band': upper_band,
        'max_active_positions': max_active_positions,
    }

start = perf_counter()
rows = list()
for window in [1, 3, 5, 10, 15, 20, 40, 60]:
    for strategy_type in ['reversal', 'momentum']: 
        for lower_band in [1, 2, 3, 4]: 
            for upper_band in [1, 2, 3, 4]: 
                for max_active_positions in [5, 20]:
                    print('Simulating', '...', window, strategy_type, upper_band, lower_band, max_active_positions)
                    row = run_simulation(
                        window, 
                        strategy_type,
                        upper_band, 
                        lower_band,
                        max_active_positions
                    )
                    rows.append(row)
df = pd.DataFrame(rows)
stop = perf_counter()
print('elapsed time:', stop-start, 'seconds\n') #elapsed time: 4241.182944779997 seconds = ~70 minutes 

# Save to csv 
df.to_csv('/Users/ellenyu/vqti/src/optimization.csv')

#%%
# load grid serach df 
df = pd.read_csv('optimization.csv')

# Check for nulls 
df.isnull().sum()

# Max cagr
df.cagr.max() #0.086376996717282 

#%%
# Visualize the results

# Scatter plot 
for i in ['window_length', 'strategy_type', 'lower_band', 'upper_band', 'max_active_positions']:
    plt.scatter(x=df[i], y=df['cagr'])
    plt.ylabel('cagr')
    plt.xlabel(i)
    plt.title('{} and CAGR'.format(i))
    plt.show()

# Box plot 
for i in ['window_length', 'strategy_type', 'lower_band', 'upper_band', 'max_active_positions']:
    boxplot = df.boxplot(column=['cagr'], by=i)
    boxplot.plot()
    plt.show()
    
# Quick observations

## window length 
# for the majority of window lengths, the median cagr is 0 which means 50% of cagrs are negative, etc. 
# lower window lengths have tighter cagr ranges 
# based on this information, I would choose higher window lengths e.g. 40 of 60 because of higher probability of upside + tighter range 

## strategy type 
# it was better to have a reversal strategy + cci during this period than momentum strategy 

## lower band 
# lower band value of 4 seems to perform better, on average 

## upper band 
# upper band value of 4 seems to perform better, on average  

## max active positions 
# max active positions of 20 seems to be better than 5 

#%% 
## lower band + reversal 
df_reversal = df.query("strategy_type =='reversal'")

boxplot = df_reversal .boxplot(column=['cagr'], by=df_reversal['upper_band'])
boxplot.plot()
plt.title('')
plt.suptitle('')
plt.ylabel('')
plt.xlabel('')
plt.show()

# I could go for a reversal strategy + lower_band =4 or lower_band = 1