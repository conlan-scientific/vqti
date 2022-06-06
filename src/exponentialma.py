from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
import unittest

# still fixing these
@time_this
def expmovingaverage(series, m: int=5):
    weights = np.exp(np.linspace(-1,0,m))
    weights /= weights.sum()
    return np.convolve(series, weights)[m-1:len(series)]

@time_this
def pandas_ema(series, m: int=5):
    return series.ewm(span=m, adjust=False).mean()

@time_this
def calculate_ema(prices, days, smoothing=2):
    ema = [sum(prices[:days]) / days]
    for price in prices[days:]:
        ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
    return ema

if __name__ == '__main__':
	
    df = load_eod('AWU')
    
    print(expmovingaverage(df.close,5))
    print(pandas_ema(df.close,5))
    print(calculate_ema(df.close,5,2))