#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:32:54 2022

This simulator code can be improved upon but, for now use what's here.'

@author: ellenyu

"""

import pandas as pd 

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
        
        # Show progress
        print(date, ' finished')
    
    return cash_dict, portfolio_dict, equity_dict
