from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

def pure_python_sma(close: List[float], m: int=10) -> List[float]:
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
def pure_python_rolling_std_dev(close: List[float], m: int=40) -> List[float]:

	sma = pure_python_sma(close, m)

	numerator_cusum = [None] * m
	accum = 0
	for i in range(m, len(close)):
		accum += (close[i] - sma[i])**2
		numerator_cusum.append(accum)

	numerator_rolling_cusum = [None] * (2*m)
	for i in range(2*m, len(close)):
		value = numerator_cusum[i] - numerator_cusum[i-m]
		numerator_rolling_cusum.append(value)

	rolling_std = [None] * (2*m)
	for i in range(2*m, len(close)):
		value = (numerator_rolling_cusum[i] / (m - 1)) ** 0.5
		rolling_std.append(value)

	return rolling_std

if __name__ == '__main__':
	
	df = load_eod('AWU')
	closes_list = df.close.tolist()

	sma = pure_python_sma(closes_list, m=20)
	rolling_std_dev = pure_python_rolling_std_dev(closes_list, m=20)

	df = df.assign(sma=pd.Series(sma, index=df.index))
	df = df.assign(
		upper_band=df.sma + 2 * pd.Series(rolling_std_dev, index=df.index),
		lower_band=df.sma - 2 * pd.Series(rolling_std_dev, index=df.index)
	)
	df[['close', 'sma', 'upper_band', 'lower_band']].iloc[-400:-200].plot()
	plt.show()


