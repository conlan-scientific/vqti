from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

@time_this
def get_the_cumsum(close: pd.Series) -> pd.Series:
	return close.cumsum()

@time_this
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

def test_pure_python_sma():
	ground_truth_result = [None, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
	test_result = pure_python_sma([1,2,3,4,5,6,7,8,9,10], m=2)
	assert len(ground_truth_result) == len(test_result)
	for i in range(len(ground_truth_result)):
		assert ground_truth_result[i] == test_result[i]

@time_this
def pandas_sma(close: pd.Series, m: int=10) -> pd.Series:
	"""
	This is an O(n) algorithm
	"""
	return close.rolling(m).mean()

@time_this
def pandas_sma_v2(close: pd.Series, m: int=10) -> pd.Series:
	"""
	This is an O(n) algorithm
	"""
	accum = close.cumsum()
	delta_accum = accum - accum.shift(m)
	return delta_accum / m

@time_this
def numpy_sma(close: np.ndarray, m: int=10) -> pd.Series:
	"""
	This is an O(n) algorithm
	"""
	accum = close.cumsum()
	delta_accum = accum[m:] - accum[:-m]
	return delta_accum / m

if __name__ == '__main__':
	
	df = load_eod('AWU')
	print(df)

	# df.close.plot()
	# plt.show()

	# result = get_the_cumsum(df.close)
	result = pure_python_sma(df.close.tolist())
	result = pandas_sma(df.close)
	result = pandas_sma_v2(df.close)
	result = numpy_sma(df.close.values)






