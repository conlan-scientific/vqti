from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List 

# NOTE: I got it to compile by changing "list" type hints to "List"
# "List" is the type hint and "list" is the list constructor.

def up_down_factors(close: List[float]):
    upList = []
    downList = []
    upList.append(close[0])
    downList.append(close[0])
    for x in range(1, len(close)):
        if (close[x] - close[x-1]) > 0:
            upList.append(close[x] - close[x-1]) # This is a first order calc
            downList.append(0)
        else:
            upList.append(0)
            downList.append(close[x-1] - close[x]) # This is a first order calc
    return upList, downList


def pure_python_smma(close: List[float], n: int = 10) -> List[float]:
    prevSmma = []
    initialSmma = sum(close, n)/n
    prevSmma.append(initialSmma)

    for x in range(1, n):
        prevSum = prevSmma[x - 1] * n
        newSmma = (prevSum - prevSmma[x - 1] + close[x])/n
        prevSmma.append(newSmma)
    
    return prevSmma

def relative_strength_index_calculation(rsf: int) -> int:
    # This forces the range into 0 to 100
    return (100 - (100/(1 + rsf)))

# TODO: The return value of this function should be List[float]
@time_this
def pure_python_relative_stength_index(close: List[float], n: int = 10) -> List[float]:
    upList, downList = up_down_factors(close) # Units are change-in-dollars
    upSmmaList = pure_python_smma(upList, n) # Units are change-in-dollars
    downSmmaList = pure_python_smma(downList, n) # Units are change-in-dollars

    # TODO: The relative strength factor should be a List[float] of equal length
    # to the input. It is the element-wise ratio of SMMA-up and SMMA-down.
    result = []
    for x in range(n):
        # This is a ratio, range is 0 to infinity
        relative_strength_factor = upSmmaList[x] / downSmmaList[x]
        current_RSI = relative_strength_index_calculation(relative_strength_factor)
        result.append(current_RSI)
    
    assert (len(close) == len(result))

    return result

    # TODO: The result should be a List[float] of equal length to the input.

def pandas_relative_strength_index(close: pd.Series, n: int = 10) -> List[float]:
    
    # TODO: The pandas version looks a lot different, because the optimal pandas
    # approach will rely on vectorized operations.
    # 
    # Here is a hint.
    delta = close - close.shift(1)
    up_series = delta.clip(lower = 0)
    delta2 = close.shift(1) - close
    down_series = delta2.clip(lower = 0)
    
    #according to wikipedia, you use either use a smooth moving average or exponential moving average, unsure how to do either via pandas
    up_series_ema = up_series.ewm(span = 30, adjust = False).mean()
    down_series_ema = down_series.ewm(span = 30, adjust = False).mean()

    relative_strength_factor = up_series_ema.div(down_series_ema)
    return relative_strength_factor.rolling(window = n).apply(lambda x: relative_strength_index_calculation(x))

def rsi_signal_line_calculation(close: List[float], n : int = 10) -> List[float]:
    calculation_list = pure_python_relative_stength_index(close, n)
    result = []
    for x in calculation_list:
        if (x < 30):
            result.append(1)
        if (x > 70):
            result.append(-1)
        else:
            result.append(0)
    return result

if __name__ == '__main__':


    df = load_eod('AWU')
    result = pure_python_relative_stength_index(df.close)

    # Thresholds of 30 and 70 are standardized
    # Thresholds in general are standardized

