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
    
    print(df.iloc[:40])

    

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
    
    # plt.grid(True, alpha = 0.3)
    # plt.plot(df.iloc[-252:]['close'], label='close')
    # plt.plot(df.iloc[-252:]['hull_ma'], label='hma')
    # plt.plot(df.iloc[-252:]['weighted_ma'], label='wma')
    # plt.plot(df.iloc[-252:]['signal'] * 100, label='signal')
    # plt.legend(loc=2)
    # plt.show()
     
    starting_cash = 10000
    shares_to_buy = starting_cash / df.close.iloc[0]
    shares_to_buy = int(shares_to_buy)
    money_to_spend = shares_to_buy * df.close.iloc[0]
    starting_cash -= money_to_spend
    portfolio_value = shares_to_buy * df.close
    cash = pd.Series(starting_cash, index=df.index)
    equity_series = cash + portfolio_value
    calculate_sharpe_ratio(equity_series)
    



