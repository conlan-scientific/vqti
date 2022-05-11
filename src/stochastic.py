from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

# The Stochastic Oscillator is a momentum indicator that shows the location of the close relative to the high-low range over a set number of periods.
# K = (closing price - low) / (high - low) x 100
# D = (k1 + k2 + k3 ....) / N

@time_this
def pandas_stochastic(close: pd.series) -> float:

	low = close.rolling(14).min()

	high = close.rolling(14).max()

	close_price = close.iloc[-1]

	k = (close_price - low) / (high - low) * 100

	return k.rolling(3).mean() 


@time_this
def numpy_stochastic(close: np.ndarray) -> float:

	low = np.nanmin(close)

    high = np.nanmax(close)

	close_price = close.iloc[-1]

	k = (close_price - low) / (high - low) * 100

	return k.rolling(3).mean() 
 



if __name__ == '__main__':
	
	df = load_eod('AWU')
	print(df)

	# df.close.plot()
	# plt.show()

	# result = get_the_cumsum(df.close)
	result = pandas_stochastic(df.close)