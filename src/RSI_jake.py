from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List 

# NOTE: I got it to compile by changing "list" type hints to "List"
# "List" is the type hint and "list" is the list constructor.


# TODO: The return value of this function should be List[float]
@time_this
def pure_python_relative_strength_index(close: List[float], n: int = 10) -> List[float]:
    upList = close.clip(lower = 0)
    downList = -1 * close.clip(upper = 0)
    upEwmList = upList.ewm(com = n - 1, adjust = True, min_periods = n).mean() # Units are change-in-dollars
    downEwmList = downList.ewm(com = n - 1, adjust = True, min_periods = n).mean() # Units are change-in-dollar
    
    rsf = upEwmList / downEwmList

    rsi = 100 - (100/(1+rsf))

    assert (len(close) == len(rsi))

    return rsi

    # TODO: The result should be a List[float] of equal length to the input.

# def pandas_relative_strength_index(close: pd.Series, n: int = 10) -> List[float]:
    
#     # TODO: The pandas version looks a lot different, because the optimal pandas
#     # approach will rely on vectorized operations.
#     # 
#     # Here is a hint.
#     delta = close - close.shift(1)
#     up_series = delta.clip(lower = 0)
#     delta2 = close.shift(1) - close
#     down_series = delta2.clip(lower = 0)
    
#     #according to wikipedia, you use either use a smooth moving average or exponential moving average, unsure how to do either via pandas
#     up_series_ema = up_series.ewm(span = 30, adjust = False).mean()
#     down_series_ema = down_series.ewm(span = 30, adjust = False).mean()

#     relative_strength_factor = up_series_ema.div(down_series_ema)
#     return relative_strength_factor.rolling(window = n).apply(lambda x: relative_strength_index_calculation(x))

# def rsi_signal_line_calculation(close: List[float], n : int = 10) -> List[float]:
#     calculation_list = pure_python_relative_strength_index(close, n)
#     result = []
#     for x in calculation_list:
#         if (x < 30):
#             result.append(1)
#         if (x > 70):
#             result.append(-1)
#         else:
#             result.append(0)
#     return result

if __name__ == '__main__':


    df = load_eod('AWU')
    result = pure_python_relative_strength_index(df.close)

    # Thresholds of 30 and 70 are standardized
    # Thresholds in general are standardized

