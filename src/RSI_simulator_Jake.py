from RSI_jake import pure_python_relative_stength_index, signal_line_calculation
from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List 

#get all data somehow, not sure :(
prices_df = get_all_data()

for (col_name, colval) in prices_df.iteritems():
    currentColRSI = pure_python_relative_stength_index(colval, len(colval))
    prices_df[col_name] = currentColRSI

for (col_name, colval) in prices_df.iteritems():
    currentColSignalLine = signal_line_calculation(colval)
    prices_df[col_name] = currentColSignalLine

max_assets = 20
df_index = prices_df.index
cash = starting_cash = 100000
portfolio: List[str] = list()
equity_curve = dict()
original_price_df = get_all_data()


for date in df_index:
    signals = prices_df.loc[date]

    stocks_im_going_to_buy: List[str] = signals[signals == 1].index.tolist()
    stocks_im_going_to_sell: List[str] = signals[signals == -1].index.tolist()
    stocks_im_holding: List[str] = portfolio

    for stock in stocks_im_going_to_sell:
        #how do we determine how mnay of each stock is in the portfolio, wouldn't a dictionary work better than a list?
        if stock in portfolio:
            portfolio.pop(stock)
            stocks_im_holding.pop(stock)
            cash += original_price_df[date][stock]
        
    for stock in stocks_im_going_to_buy:
        cash_to_spend = cash / (max_assets - len(stocks_im_holding))
        #I'm not sure how we're dividing up money between stocks, even for all 20 stocks?
        if stock not in portfolio:
            if len(stocks_im_holding) < max_assets:
                portfolio.append(stock)
                stocks_im_holding.append(stock)
           # else:
                #how to deal with overflow???

    current_portfolio_value = 0
    for stock in portfolio:
        current_portfolio_value += original_price_df[date][stock]

    equity_curve[date] = cash + current_portfolio_value
#still need equity curve and sharpe ratio