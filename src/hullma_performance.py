from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

from vqti.performance import (
	calculate_cagr,
	calculate_annualized_volatility,
	calculate_sharpe_ratio,
)

from hullma import (
    numpy_matrix_hma,
    numpy_matrix_wma
)

def hma_trend(series: pd.Series) -> pd.Series:
    trend = np.sign(series - series.shift(1))
    return trend

def hma_trend_signal(series: pd.Series) -> pd.Series:
    signal = np.where(series > series.shift(1), 1, 0)
    signal = np.where(series < series.shift(1), -1, signal)
    return signal
    
    
if __name__ == '__main__':

    df = load_eod('AWU')
    
    # weighted_ma = numpy_matrix_wma(np.array(df.close), 20)
    # df = df.assign(weighted_ma=weighted_ma)

    hull_ma = numpy_matrix_hma(np.array(df.close), 16)
    df = df.assign(hull_ma=hull_ma)
    
    df['trend'] = hma_trend(df.hull_ma)
    df['signal'] = hma_trend_signal(df.trend)
    # df['signal'] = np.where(df['trend'] < df['trend'].shift(1), -1, 0)
    # df['signal'] = np.where(df['trend'] > df['trend'].shift(1), 1, df['signal'])
    
    print(df.iloc[:40])
    # print(df)
    
    plt.grid(True, alpha = 0.3)
    plt.plot(df.iloc[-252:]['close'], label='close')
    plt.plot(df.iloc[-252:]['hull_ma'], label='hma')
    # plt.plot(df.iloc[-252:]['weighted_ma'], label='wma')
    plt.plot(df.iloc[-252:]['signal'] * 100, label='signal')
    plt.legend(loc=2)
    plt.show()
     
    starting_cash = 10000
    shares_to_buy = starting_cash / df.close.iloc[0]
    shares_to_buy = int(shares_to_buy)
    money_to_spend = shares_to_buy * df.close.iloc[0]
    starting_cash -= money_to_spend
    portfolio_value = shares_to_buy * df.close
    cash = pd.Series(starting_cash, index=df.index)
    equity_series = cash + portfolio_value
    calculate_sharpe_ratio(equity_series)
    



