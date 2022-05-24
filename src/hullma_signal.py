from vqti.load import EOD_DIR, load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict
import os
import glob

from vqti.performance import (
	calculate_cagr,
	calculate_annualized_volatility,
	calculate_sharpe_ratio,
)

from hullma import (
    numpy_matrix_hma,
    numpy_matrix_wma
)

def hma_trend_signal(series: pd.Series, m: int=10) -> pd.Series:
    hull_ma = pd.Series(numpy_matrix_hma(series.values, m))
    trend = np.sign(hull_ma - hull_ma.shift(1))
    signal = np.where(trend > trend.shift(1), 1, 0)
    signal = np.where(trend < trend.shift(1), -1, signal)
    return signal


"""
Standardization ideas (also signal line ideas)
+ Using trends and signs(DONE)
+ Crossover of two different HMAs
+ Crossover of an HMA and the price
+ Difference between two HMAs divided by volatility (z-score units) (similar to MACD)
"""
    
    
if __name__ == '__main__':

    df = load_eod('AWU')
    df['signal'] = hma_trend_signal(df.close, 16)
    # print(df.iloc[:40])
    # plt.grid(True, alpha = 0.3)
    # plt.plot(df.iloc[-252:]['close'], label='close')
    # plt.plot(df.iloc[-252:]['hull_ma'], label='hma')
    # plt.plot(df.iloc[-252:]['weighted_ma'], label='wma')
    # plt.plot(df.iloc[-252:]['signal'] * 100, label='signal')
    # plt.legend(loc=2)
    # plt.show()
    

    os.chdir('data\eod')
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))] # creates a list of symbols.csv
    stock_symbols = [i.replace('.csv', '') for i in all_filenames] # removes .csv from end of symbols
    path = EOD_DIR 
    all_files = glob.glob(path + '/*.csv') # creates a list of file paths to each csv file in eod directory
    file_dict = dict(zip(stock_symbols, all_files)) # creates a dictionary of 'stock symbol' : 'file path'
    
    # from algorithmic trading repository
    def combine_columns(filepaths_by_symbol: Dict[str, str]) -> pd.DataFrame:

        data_frames = [
        pd.read_csv(
            filepath, 
            index_col='date', 
            usecols=['date', 'close'], 
            parse_dates=['date'],
            dtype='float64'
        ).rename(
            columns={
                'date': 'date', 
                'close': symbol,
            }
        ) for symbol, filepath in filepaths_by_symbol.items()
        ]
        return pd.concat(data_frames, sort=True, axis=1)    

    prices_df = combine_columns(file_dict)
    print(prices_df)
    
    #calculate the signals
    def calculate_signal_df(dataframe: pd.DataFrame, m: int=10) -> pd.DataFrame:
        return dataframe.apply(lambda x: hma_trend_signal(x, m), axis=0)
    
    signal_df = calculate_signal_df(prices_df, 16)
    print(signal_df)
    assert prices_df.index.equals(signal_df.index)
    
    max_assets = 5
    dt_index = prices_df.index
    starting_cash = int(100000)
    portfolio: List[str] = list()
    portfolio_value = 0
    cash = starting_cash
    equity_curve = {}
    stocks_im_holding = {}
    # Pretend you are walking through time trading over the course of ten years
    for date in signal_df.index:
        signals = signal_df.loc[date] 
        stocks_im_going_to_buy: List[str] = signals[signals == 1].index.tolist()
        stocks_im_going_to_sell: List[str] = signals[signals == -1].index.tolist()
        
        # Mess with your cash and portfolio to sell stocks
        if not stocks_im_holding:
            pass
        else:
            for stock in stocks_im_going_to_sell:
                if stock in stocks_im_holding:
                    shares_sold = prices_df.loc[date][f'{stock}'] * stocks_im_holding.get(stock)
                    cash += shares_sold
                    stocks_im_holding.pop(stock)
                else:
                    pass

        if (len(stocks_im_holding) < max_assets) and (cash > 0):
            cash_to_spend = cash / (max_assets - len(stocks_im_holding))
            for stock in stocks_im_going_to_buy:
                if (len(stocks_im_holding) < max_assets) and (cash > 0):
                    # This is the "compound your gains and losses" approach to cash management
                    # Also called the "fixed number of slots" approach
                    shares_bought = cash_to_spend / prices_df.loc[date][f'{stock}']
                    cash -= shares_bought * prices_df.loc[date][f'{stock}']
                    portfolio_value += shares_bought * prices_df.loc[date][f'{stock}']
                    portfolio.append(stock)
                    stocks_im_holding[f'{stock}'] =  shares_bought
        
        equity_curve[f'{date}'] = cash + portfolio_value
                
    equity_curve_df = pd.Series(equity_curve, name = 'total_equity', dtype = float)
    equity_curve_df.index.name = 'Date'
    # Plot the equity curve
    print(equity_curve_df.head())
    # plt.plot(equity_curve_df)
    # plt.show()
    # Measure the sharpe ratio





