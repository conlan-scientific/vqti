from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

def pure_python_sma(close: List[float], m: int=10) -> List[float]:
	"""
	This is an O(n) algorithm
	"""
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

@time_this
def pandas_sma_v2(close: pd.Series, m: int=10) -> pd.Series:
	"""
	This is an O(n) algorithm
	"""
	accum = close.cumsum()
	delta_accum = accum - accum.shift(m)
	return delta_accum / m

@time_this
def pandas_macd(close: pd.Series, n1: int = 2, n2: int = 3) -> pd.Series:
    assert n1 < n2
    sma1 = pandas_sma_v2(close, n1)
    sma2 = pandas_sma_v2(close, n2)
    return sma1.subtract(sma2)

@time_this
def numpy_sma(close: np.ndarray, m: int=10) -> np.ndarray:
	"""
	This is an O(n) algorithm
	"""
	accum = close.cumsum()
	delta_accum = accum[m:] - accum[:-m]
	return delta_accum / m

@time_this
def numpy_macd(close: np.ndarray, n1: int = 2, n2: int = 3) -> np.ndarray:
    assert n1 < n2
    sma1 = numpy_sma(close, n1)
    sma1 = np.insert(sma1, 0, [np.nan] * n1)
    sma2 = numpy_sma(close, n2)
    sma2 = np.insert(sma2, 0, [np.nan] * n2)
    return sma1 - sma2

df = load_eod('AWU')
# print('Loaded Data:')
# print(df)
# print('-----------')
# print('Pure Python MACD')
print(macd_python(df.close.tolist(), n1=2, n2=3))
# print('-----------')
print(pandas_macd(df.close, n1=2, n2=3))
print(numpy_macd(df.close.values, n1=2, n2=3))

# Positive test case
input_data = [1,2,3,4,5,6,7,8,9,10]
trust_case_data = [None, None, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

test_case = macd_python(input_data, n1=1, n2=2)
truth_case = trust_case_data
assert test_case == truth_case, 'Test failed.'

test_case = pandas_macd(pd.Series(input_data), n1=1, n2=2)
truth_case = pd.Series(trust_case_data)
assert test_case == truth_case, 'Test failed.'

test_case = numpy_macd(np.ndarray(input_data), n1=1, n2=2)
truth_case = np.array(trust_case_data)
assert test_case == truth_case, 'Test failed.'

# Negative test case
try:
    macd_python(input_data, n1=-2, n2=4)
except AssertionError as e:
    assert e.args[0] == 'Window must be positive.', 'Uncaught error.'

