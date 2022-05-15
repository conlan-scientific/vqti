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
            upList.append(close[x] - close[x-1])
            downList.append(0)
        else:
            upList.append(0)
            downList.append(close[x-1] - close[x])
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


# TODO: The return value of this function should be List[float]
@time_this
def pure_python_relative_stength_index(close: List[float], n: int = 10) -> float:
    upList, downList = up_down_factors(close)
    upSmmaList = pure_python_smma(upList, n)
    downSmmaList = pure_python_smma(downList, n)

    # TODO: The relative strength factor should be a List[float] of equal length
    # to the input. It is the element-wise ratio of SMMA-up and SMMA-down.
    upSmmaTotal = sum(upSmmaList)
    downSmmaTotal = sum(downSmmaList)
    relativeStrengthFactor = upSmmaTotal/downSmmaTotal

    # TODO: The result should be a List[float] of equal length to the input.
    result = 100 - (100/ (1 + relativeStrengthFactor))

    return result

def pandas_relative_strength_index(close: List[float], n: int = 10) -> float:
    #it's the exact same but I think there is a way to do the smoothed moving average with pandas, not sure how though
    
    # TODO: The pandas version looks a lot different, because the optimal pandas
    # approach will rely on vectorized operations.
    # 
    # Here is a hint.
    delta = close - close.shift(1)
    up_series = delta.clip(lower=0)
    # ... 
    return 10


if __name__ == '__main__':


    df = load_eod('AWU')
    result = pure_python_relative_stength_index(df.close)

