from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List 

# NOTE: I got it to compile by changing "list" type hints to "List"
# "List" is the type hint and "list" is the list constructor.


def relative_strength_index(close: List[float], n: int = 14) -> List[float]:
    delta = close.diff()

    upList = delta.clip(lower = 0)
    downList = -1 * delta.clip(upper = 0)

    upEwmList = upList.ewm(com = n - 1, adjust = True, min_periods = n).mean() # Units are change-in-dollars
    downEwmList = downList.ewm(com = n - 1, adjust = True, min_periods = n).mean() # Units are change-in-dollar
    
    rsf = upEwmList / downEwmList

    rsi = 100 - (100/(1+rsf))

    assert (len(close) == len(rsi))

    return rsi


def rsi_signal_line_calculation(close: List[float], n : int = 14) -> List[float]:
    calculation_list = relative_strength_index(close, n)
    signal = calculation_list
    crossUp = False
    crossDown = False
    for x in range(len(calculation_list)):
        if (calculation_list[x] >= 80 and not crossUp):
            signal[x] = 1
            crossUp = True
            crossDown = False
        elif (calculation_list[x] <= 20 and not crossDown):
            signal[x] = -1
            crossDown = True
            crossUp = False
        elif (calculation_list[x] > 20 and calculation_list[x] < 80):
            signal[x] = 0
            crossDown = False
            crossUp = False
        else:
            signal[x] = 0
    #signal = np.where(calculation_list > 70, 1, 0) #crossover based, whne it crosses over 70 the first time 
    #signal = np.where(calculation_list < 30, -1, signal)
    return signal

if __name__ == '__main__':


    df = load_eod('AWU')
    result = relative_strength_index(df.close, 14)

    # Thresholds of 30 and 70 are standardized
    # Thresholds in general are standardized

