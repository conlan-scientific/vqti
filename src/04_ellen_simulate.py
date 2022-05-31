#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 08:54:17 2022

Explored the performance of cci via two simulators. 

Pending: 
    * double check if quick simulator is incorrect because the results 
    (equiy curve, cagr, sharpe ratio) are horrible.  

@author: ellenyu

"""
import pandas as pd
import numpy as np
from load_ellen import * 
from cci_ellen import * 
from signal_ellen import * 
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
def quick_simulator (price_df: pd.DataFrame, signal_df: pd.DataFrame, cash: int = 100_000, num_partitions: int = 20):
    
    # Double check signal df which is created off price df has the same index as price df 
    assert price_df.index.equals(signal_df.index), "Indices do not equal"

    # Initiate dictionaries 
    shares_dict: Dict [stock: str, num_share: int] = {}
    stocks_dict: Dict [date, stocks: List[str]] = {}
    cash_dict: Dict [date, cash: int] = {}
    portfolio_dict: Dict [date, value: int ]= {}
    equity_dict: Dict [date, value: int ]= {}
    
    # Split my money into fixed number of partitions                    
    cash_to_spend = cash / num_partitions
    
    # Start simulator code 
    for date in price_df.index:
        #print(type(date)) #<class 'pandas._libs.tslibs.timestamps.Timestamp'>
        for stock in price_df.columns:
            signal = signal_df.loc[date, stock]
            #print(type(signal)) #<class 'numpy.longlong'>
            price = price_df.loc[date, stock] 
            # print(type(price)) #<class 'numpy.float64'> 
            if signal == 1 and cash > cash_to_spend: 
                #Buy 
                cash -= cash_to_spend
                shares = cash_to_spend / price
                shares_dict[stock] = shares 
            elif signal == -1 and stock in list(shares_dict.keys()):
                #Sell 
                stock_value = price * shares_dict[stock]
                cash += stock_value
                shares_dict.pop(stock)
            else: 
                #Do nothing 
                pass
       
        # Keep a few records 
        cash_dict [date] = cash
        stocks_dict[date] = list(shares_dict.keys())
        value = 0 
        for stock in stocks_dict[date]:
            value += shares_dict[stock] * price_df.loc[date, stock]
        portfolio_dict [date] = value
        equity_dict [date] = cash_dict[date] + portfolio_dict[date]
    
    return cash_dict, portfolio_dict, equity_dict

#%%
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
# Method - Class 
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
