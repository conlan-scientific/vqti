from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

def new_python_sma(close: List[float], m: int=10) -> List[float]:
    result = [None] * (m-1)
    for i in range(len(close)-(m-1)):
        result.append(sum(close[i:i+m])/m)
    return result
def pure_python_sma(close: List[float], m: int=10) -> List[float]:
	"""
	This is an O(n) algorithm
	"""
	assert m >= 1, 'Window must be positive.'
	result = [None] * (m-1)
	accum = sum(close[:m])
	result.append(accum / m)
	for i in range(m, len(close)):
		accum += close[i]
		accum -= close[i-m]
		result.append(accum / m)
	return result
@time_this
def macd_python(close: List[float], n1: int = 2, n2: int = 3) -> List[float]:
    """
    O(N)(3) Algorithm (?)
    """
    assert n1 < n2
    sma1 = pure_python_sma(close, n1)
    sma2 = pure_python_sma(close, n2)
    result = []
    for i in range(len(sma1)):
        if sma1[i] == None:
            result.append(None)
        elif sma2[i] == None:
            result.append(sma1[i])
        else:
            result.append(sma1[i] - sma2[i])
    return result

input_data = [2,10,4,1,1,4,7,15,2,4]
test_macd = macd_python(input_data, 2, 3)

def macd_signal(macd: List[float]) -> List[float]:
    result = []
    for i in range(len(macd) - 1):
        if macd[i] == None:
            result.append(None)
        elif macd[i] > 0 and macd[i+1] < 0:
            result.append(-1)
        elif macd[i] < 0 and macd[i+1] > 0:
            result.append(1)
        else:
            result.append(0)
    #add an additional zero value for the final close value
    result.append(0)
    return result
test_signal = macd_signal(test_macd)
print(macd_signal(test_macd))
df = load_eod('AWU')
real_test_macd = macd_python(df.close.tolist(), n1=12, n2=26)
real_test_signal = macd_signal(real_test_macd)
#TODO: get axis scaling correct, so that the signal line is actually visible
plt.plot(df.close.tolist(), label='Close Price', color = 'green')
plt.plot(real_test_macd, label = 'MACD', color = 'red')
plt.plot(real_test_signal, label = 'Signal', color = 'blue')
plt.legend(loc='upper left')
plt.show()