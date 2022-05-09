from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

@time_this
def macd_python(close: List[float], n1: int = 2, n2: int = 3) -> List[float]:
    """
    O(N)(3) Algorithm (?)
    """
    assert n1 < n2
    sma1 = [None] * (n1 - 1)
    accum1 = sum(close[:n1])
    sma1.append(accum1 / n1)
    for i in range(n1, len(close)):
        accum1 += close[i]
        accum1 -= close[i - n1]
        sma1.append(accum1 / n1)
    sma2 = [None] * (n2 - 1)
    accum2 = sum(close[:n2])
    sma2.append(accum2 / n2)
    for i in range(n2, len(close)):
        accum2 += close[i]
        accum2 -= close[i - n2]
        sma2.append(accum2 / n2)
    result = []
    for i in range(len(sma1)):
        if sma1[i] == None:
            result.append(None)
        elif sma2[i] == None:
            result.append(sma1[i])
        else:
            result.append(sma1[i] - sma2[i])
    return result
print(macd_python([1,2,3,4,5,6,7,8,9,10], n1=2, n2=3))


def pandas_macd(close: pd.Series, n1: int = 2, n2: int = 3):
    assert n1 < n2
    sma1 = (close.cumsum() - close.cumsum().shift(n1)) / n1
    sma2 = (close.cumsum() - close.cumsum().shift(n2)) / n2
    return sma1 - sma2



def numpy_macd(close: np.ndarray, n1: int = 2, n2: int = 3):
    assert n1 < n2
    sma1 = (close.cumsum()[n1:] - close.cumsum()[:-n1]) / n1
    sma2 = (close.cumsum()[n2:] - close.cumsum()[:-n2]) / n2
    return sma1 - sma2
    """
    to do here:
    figure out how to deal with dimensional mismatch during subtraction
    """


df = load_eod('AWU')
print('Loaded Data:')
print(df)
print('-----------')
print('Pure Python MACD')
print(macd_python(df.close.tolist(), n1=2, n2=3))
print('-----------')
print(pandas_macd(df.close, n1=2, n2=3))
print(numpy_macd(df.close.values, n1=2, n2=3))