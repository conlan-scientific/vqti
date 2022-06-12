#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 08:09:03 2022

Second python script running through 6305 tickers and 64 parameter sets stopped in the middle
due to 'assert self.cash >= 0, 'Spent cash you do not have.'' error. It did completed 35 sets 
and spent 5.3 hours (while I was running another python script and using the internet) to do so. 

Notice the nan in percent returns and cagr, the spent cash you do not have, not to mention the 
jensen alpha function I commented out. Clearly, I have to spend time in troubleshooting the simulator 
(which I will do so on the smaller set of data). Update: I believe this is complete as of June 2022'
 
@author: ellenyu

"""

import pandas as pd
from load_ellen import * 
from cci_ellen import * 
from signal_ellen import * 
from pypm import metrics, signals, data_io, simulation
from typing import List, Dict, Any
from time import perf_counter


# Load in prices_df - extended tickers
price_df = pd.read_csv('/Users/ellenyu/Desktop/UVA MSDS/Capstone/Coding/prices.csv', parse_dates = ['date'], index_col='date')
price_df = price_df[['ticker', 'close_split_adjusted']]
price_df = price_df.pivot_table(index = 'date', columns='ticker', values='close_split_adjusted')
#preference = pd.DataFrame(0, index=prices.index, columns=prices.columns)
preference_df = price_df.apply(metrics.calculate_rolling_sharpe_ratio, axis=0)

#%%
## Following Chris' model 
def run_simulation (window: int, strategy_type: str, lower_band: int, upper_band: int, max_active_positions: int) -> Dict[str, Any]: 
    
    # Generate signals
    indicator_df = price_df.apply(lambda col: python_ccimodified_loop(col.tolist(), window=window), axis=0)
    signal_df = cci_signals_generator(series=indicator_df, strategy_type=strategy_type, upper_band=upper_band, lower_band=lower_band)
    
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
    simulator.simulate(price = price_df, signal = signal_df, preference = preference_df)


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
        #'jensens_alpha': portfolio_history.jensens_alpha_v2,
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

#%%
start = perf_counter()
rows = list()
for window in [30, 50]:
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
optimization_df = pd.DataFrame(rows)
stop = perf_counter()
print('elapsed time:', stop-start, 'seconds\n') #elapsed time: 19009.515875532 seconds = 5.280421076536667 hours

# Save to csv 
optimization_df.to_csv('/Users/ellenyu/vqti/src/optimization_6305_35_2.csv')

#%%
# load grid serach df 
optimization_df = pd.read_csv('/Users/ellenyu/vqti/src/optimization_6305_35_2.csv')

# Check for nulls 
optimization_df.isnull().sum()

# Max cagr
optimization_df.cagr.max() # nan

# Visualize the results
# Box plot 
for i in ['window_length', 'strategy_type', 'lower_band', 'upper_band', 'max_active_positions']:
    boxplot = optimization_df.boxplot(column=['cagr'], by=i)
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

## lower band + reversal 
df_reversal = optimization_df.query("strategy_type =='reversal'")

boxplot = df_reversal .boxplot(column=['cagr'], by=df_reversal['upper_band'])
boxplot.plot()
plt.title('')
plt.suptitle('')
plt.ylabel('')
plt.xlabel('')
plt.show()

# I could go for a reversal strategy + lower_band = 4 or lower_band = 1