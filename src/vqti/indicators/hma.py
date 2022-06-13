import pandas as pd
import numpy as np
from typing import List
# Hull Moving Average
# HMA= WMA(2*WMA(n/2) − WMA(n)),sqrt(n))
# recommended m = 4, 9, 16, 25, 49, 81

# fastest wma
def calculate_numpy_matrix_wma(values: np.ndarray, m: int=16) -> np.ndarray:
    assert m >= 1, 'Period must be a positive integer'
    assert type(m) is int, 'Period must be a positive integer'
    assert len(values) >= m, 'Values must be >= period m'
    n = values.shape[0]
    weights = []
    denom = (m * (m + 1)) / 2
    for i in range(1, m + 1):
        x = i / denom
        weights.append(x)

    weights = np.array(weights)
    # Exit early if m greater than length of values
    if m > n:
        return np.array([np.nan] * n)
    
    # Front padding of series
    front_pad = max(m - 1, 0)

    # Initialize the output array
    y = np.empty((n,))

    # Pad with na values
    y[:front_pad] = np.nan

    # Build a matrix to multiply with weight vector
    q = np.empty((n - front_pad, m))
    for j in range(m):
        q[:,j] = values[j:(j+n-m+1)]

    y[front_pad: len(values)] = q.dot(weights)

    return y

# fastest hma
def calculate_numpy_matrix_hma(values: np.ndarray, m: int=10) -> np.array:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return calculate_numpy_matrix_wma(
                (2* calculate_numpy_matrix_wma(values, int(m/2))) -\
                    (calculate_numpy_matrix_wma(values, m)), int(np.sqrt(m)))

# fastest pandas wma
def calculate_pandas_wma(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return values.rolling(m).apply(lambda x: ((np.arange(m)+1)*x).sum()/ \
                            (np.arange(m)+1).sum(), raw=True)
 
def calculate_pandas_hma(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return calculate_pandas_wma(2* calculate_pandas_wma(values, int(m/2)) - \
                        (calculate_pandas_wma(values, m)), int(np.sqrt(m)))

def _calculate_pure_python_wma(values: List[float], m: int=10)-> List[float]:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	# Building a triangular weight array
	weighting = []
	for i in range(1, m+1):
		weighting.append(i)

	# Initial values
	moving_average = [np.nan] * (m-1)
	
	# Apply weights
	for i in range(m-1, len(values)):
		the_average = np.average(values[(i-m+1):i+1], weights=weighting)
		moving_average.append(the_average)

	return moving_average


def _calculate_pure_python_hma(values: List[float], m: int=16) -> List[float]:
	"""
	This is a smoothed 0th order calculation expressed in dollars per share
	"""
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
 
	wma1 = np.array(_calculate_pure_python_wma(values, int(m/2)))
	# multiply wma1 by 2 while keeping nan values
	wma1_multiplied = [None] * (int(m/2) -1)
	for i in range((int(m/2)-1), len(wma1)):
		y = wma1[i] *2
		wma1_multiplied.append(y)
  
	wma2 = np.array(_calculate_pure_python_wma(values, m))
	# subtract wma2 from wma1 multiplied while keeping null values
	raw_hma = [0] * (m-1)
	for i in range((m-1), len(wma2)):
		raw_hma.append(np.subtract(wma1_multiplied[i], wma2[i]))
	hma= _calculate_pure_python_wma(raw_hma, int(np.sqrt(m)))
	hma[0:m] = [None] * m
	return hma

def _calculate_numpy_wma(values: np.array, m: int=10) -> np.array:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
 
	n = values.shape[0]
	weights = []
	denom = (m * (m+1)) / 2
	for i in range(1, m+1):
		x = i / denom 
		weights.append(x)

	weights = np.array(weights)
    # Exit early if m greater than length of values
	if m > n:
		return np.array([np.nan]*n)

    # Front padding of series
	front_pad = max(m-1, 0)
   
    # Initialize the output array
	y = np.empty((n,))

    # Pad with na values
	y[:front_pad] = np.nan
    
    # Compute the moving average
	for i in range(front_pad, n):
		x = values[i]
		y[i] = weights.dot(values[(i-m+1):i+1])

	return y


def _calculate_numpy_hma(values: np.ndarray, m: int=16) -> np.array:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return _calculate_numpy_wma((2* _calculate_numpy_wma(values, int(m/2))) -\
                    (_calculate_numpy_wma(values, m)), int(np.sqrt(m)))

  
def _calculate_pandas_wma_2(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	weights = []
	for i in range(1, m+1):
		weights.append(i)
	sum_weights = np.sum(weights)
	return values.rolling(window=m).apply(lambda x: np.sum(weights*x) / \
                                       sum_weights)


def _calculate_pandas_wma_3(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	weights = []
	denom = (m * (m+1)) / 2
	for i in range(1, m+1):
		x = i / denom 
		weights.append(x)
	weights = np.array(weights)
	return values.rolling(window=m).apply(lambda x: np.sum(weights*x))

