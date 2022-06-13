from vqti.load import EOD_DIR, load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict
import os
import glob
from IPython import embed as ipython_embed

from vqti.performance import (
	calculate_cagr,
	calculate_annualized_volatility,
	calculate_sharpe_ratio,
)

from hullma import (
    numpy_matrix_hma,
    numpy_matrix_wma
)

def hma_trend_signal(series: pd.Series, m: int=49) -> pd.Series:
    hull_ma = pd.Series(numpy_matrix_hma(series.values, m))
    trend = np.sign(hull_ma - hull_ma.shift(1))
    signal = np.where(trend > trend.shift(1), 1, 0)
    signal = np.where(trend < trend.shift(1), -1, signal)
    return signal

def wma_trend_signal(series: pd.Series, m: int=30) -> pd.Series:
    wma = pd.Series(numpy_matrix_wma(series.values, m))
    trend = np.sign(wma - wma.shift(1))
    signal = np.where(trend > trend.shift(1), 1, 0)
    signal = np.where(trend < trend.shift(1), -1, signal)
    return signal

def rolling_volatility(series, m):
    return series.rolling(m).std() * np.sqrt(252/m)
 
def hma_zscore(series: pd.Series, m1: int=16, m2: int=81) -> pd.Series:
    assert m1 < m2, "m1 must be less than m2"
    hma1 = pd.Series(numpy_matrix_hma(series.values, m1), index=series.index) # / series.rolling(m1).std())
    hma2 = pd.Series(numpy_matrix_hma(series.values, m2), index=series.index) # / series.rolling(m2).std())
    vol = rolling_volatility(series, m2)
    return (hma1 - hma2) / vol #volatility should be on the same length as the indicator

def hma_zscore_signal(series: pd.Series, m1: int=16, m2: int=81):
    zscore = hma_zscore(series, m1, m2)
    zscore_sign = np.sign(zscore)
    zscore_shifted_sign = zscore_sign.shift(1, axis=0)
    return zscore_sign * (zscore_sign != zscore_shifted_sign)

def hma_macd_signal(series: pd.Series, m1: int=16, m2: int=49, sig: int=9) -> pd.Series:
    assert m1 < m2, "m1 must be less than m2"
    assert sig < m1, 'signal line must be less than m1'
    hma1 = pd.Series(numpy_matrix_hma(series.values, m1), index=series.index) 
    hma2 = pd.Series(numpy_matrix_hma(series.values, m2), index=series.index)
    macd = hma1 - hma2
    macd_sig = pd.Series(numpy_matrix_hma(macd.values, sig), index=series.index)
    hist = macd - macd_sig
    hist_sign = np.sign(hist)
    hist_shifted_sign = hist_sign.shift(1, axis=0)
    return hist_sign * (hist_sign != hist_shifted_sign)

def hma_price_crossover(series: pd.Series, m: int=16):
    series = np.array(series)
    hull_ma = pd.Series(numpy_matrix_hma(series, m))
    sign = np.where(hull_ma > series, 1, 0)
    sign = pd.Series(np.where(hull_ma < series, -1, sign))
    price_crossover = np.where(sign > sign.shift(1), 1, 0)
    price_crossover = np.where(sign < sign.shift(1), -1, price_crossover)
    return price_crossover 

def hma_crossover(series: pd.Series, m1: int=16, m2: int=81)-> pd.Series:
    fast_hma = pd.Series(numpy_matrix_hma(series, m1))
    slow_hma = pd.Series(numpy_matrix_hma(series, m2))
    sign = np.sign(fast_hma - slow_hma)
    crossover = np.where(sign > sign.shift(1), 1, 0)
    crossover = np.where(sign < sign.shift(1), -1, crossover)
    return crossover

def atr(dataframe: pd.DataFrame, n: int=14,):
    assert pd.Index(['close', 'high', 'low']).isin(df.columns), 'Data frame must have high, low, close in columns'
    high_low = dataframe['high'] - dataframe['low']
    high_close = np.abs(dataframe['high'] - dataframe['close'].shift())
    low_close = np.abs(dataframe['low'] - dataframe['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(n).sum()/n
    return atr

def volatility(series: pd.Series, n: int=14):
    log_returns = pd.Series(np.log(series / series.shift(-1)))
    vol = log_returns.rolling(window=n).std()*np.sqrt(n)
    return vol


"""
Standardization ideas (also signal line ideas)
+ Using trends and signs(DONE)
+ Crossover of two different HMAs
+ Crossover of an HMA and the price
+ Difference between two HMAs divided by volatility (z-score units) (similar to MACD)
"""
    
    
if __name__ == '__main__':

    df = load_eod('AWU')
    '''
    df['pricecross'] = (hma_price_crossover(df.close, 4))
    df['vol'] = volatility(df.close, 14)
    df['fast'] = numpy_matrix_hma(df.close, 4)
    df['slow'] = numpy_matrix_hma(df.close, 16)
    df['crossover'] = hma_crossover(df.close,4,16)
    df['atr']= atr(df,14)
    df['zscore'] = hma_zscore(df.close,4,16)
    df['zsore_sig'] = hma_zscore_signal(df.close,4,16)
    '''
    df['macd_sig'] = hma_macd_signal(df.close, 16, 49, 9)
    print(df.iloc[:100])
    # plt.grid(True, alpha = 0.3)
    # plt.plot(df.iloc[-252:]['close'], label='close')
    # plt.plot(df.iloc[-252:]['hull_ma'], label='hma')
    # plt.plot(df.iloc[-252:]['weighted_ma'], label='wma')
    # plt.plot(df.iloc[-252:]['signal'] * 100, label='signal')
    # plt.legend(loc=2)
    # plt.show()
    
    # unit test for hma_trend_signal
    ## generate the signals 
    signal = hma_trend_signal(df.close)
    print("signal:", signal, '\n')
    ## find the indices where signals = 1 or -1
    signal_series = pd.Series(signal)
    signal_index = signal_series.loc[signal_series!=0].index
    print("signal_index:", signal_index, '\n')
    ## general the trends 
    hull_ma = pd.Series(numpy_matrix_hma(df.close.values, m=49))
    trend = np.sign(hull_ma - hull_ma.shift(1))
    trend = trend.fillna(0)
    print("trend:", trend, '\n')
    ## find the indices where signals = 1 or -1
    trend_index = trend.loc[trend!=trend.shift(1)].index
    trend_index = trend_index.delete([0, 1])
    print("trend_index:", trend_index, '\n')
    ##assert the two indices are equal
    assert trend_index.equals(signal_index), "Test Failed"
    assert np.array_equal(trend_index, signal_index), "Test Failed"
    
    os.chdir('data\eod') #FileNotFoundError: [Errno 2] No such file or directory: 'data\\eod'
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
    def calculate_signal_df(dataframe: pd.DataFrame, m: int=16) -> pd.DataFrame:
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
                    portfolio_value -= shares_sold
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
        if date == signal_df.index[-1]:
            for stock in stocks_im_holding: 
                shares_sold = prices_df.loc[date][f'{stock}'] * stocks_im_holding.get(stock)
                cash += shares_sold
                portfolio_value -= shares_sold
    
        equity_curve[f'{date}'] = cash + portfolio_value
    
    equity_curve_df = pd.Series(equity_curve, name = 'total_equity')
    equity_curve_df.index.name = 'Date'
    # Plot the equity curve
    print(equity_curve_df)
    # plt.plot(equity_curve_df)
    # plt.show()
    # Measure the sharpe ratio