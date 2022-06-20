import pandas as pd
import numpy as np
from typing import List
import unittest
# Hull Moving Average
# HMA = WMA(2*WMA(n/2) − WMA(n)),sqrt(n))
# recommended m = 4, 9, 16, 25, 49, 81

# fastest wma
def calculate_numpy_matrix_wma(series: pd.Series, m: int=16) -> pd.Series:
    assert m >= 1, 'Period must be a positive integer'
    assert type(m) is int, 'Period must be a positive integer'
    assert len(series) >= m, 'Values must be >= period m'
    values = np.array(series.values)
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
    wma = np.empty((n,))

    # Pad with na values
    wma[:front_pad] = np.nan

    # Build a matrix to multiply with weight vector
    q = np.empty((n - front_pad, m))
    for j in range(m):
        q[:,j] = values[j:(j+n-m+1)]

    wma[front_pad: len(values)] = q.dot(weights)

    return pd.Series(wma, index=series.index, dtype='float64', name='wma')

# fastest hma
def calculate_numpy_matrix_hma(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	hma = calculate_numpy_matrix_wma(
                (2* calculate_numpy_matrix_wma(values, int(m/2))) -\
                    (calculate_numpy_matrix_wma(values, m)), int(np.sqrt(m)))
	return pd.Series(hma.values, index=values.index, dtype='float64', name='hma')

# fastest pandas wma
def calculate_pandas_wma(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	wma = values.rolling(m).apply(lambda x: ((np.arange(m)+1)*x).sum()/ \
                            (np.arange(m)+1).sum(), raw=True)
	return pd.Series(wma.values, index=values.index, dtype='float64', name='hma')
def calculate_pandas_hma(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	hma = calculate_pandas_wma(2* calculate_pandas_wma(values, int(m/2)) - \
                        (calculate_pandas_wma(values, m)), int(np.sqrt(m)))
	return pd.Series(hma.values, index=values.index, dtype='float64', name='hma')

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
	wma = np.empty((n,))

    # Pad with na values
	wma[:front_pad] = np.nan
    
    # Compute the moving average
	for i in range(front_pad, n):
		x = values[i]
		wma[i] = weights.dot(values[(i-m+1):i+1])

	return wma


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

####### TESTING FUNCTIONS #######

class TestWMA(unittest.TestCase):
	def setUp(self):
		self.input_data = [1,2,3,4,5,6,7,8,9,10]
		self.truth_case_data = [np.nan, np.nan, np.nan, 3.0, 4.0, 
                          		5.0, 6.0, 7.0, 8.0, 9.0]

	def test_pure_python_wma(self):
		test_case = _calculate_pure_python_wma(self.input_data, m=4)
		self.assertEqual(test_case, self.truth_case_data)
	
	def test_numpy_wma(self):
		test_case = _calculate_numpy_wma(np.array(self.input_data), m=4)
		truth_case = np.array(self.truth_case_data)
		np.testing.assert_array_equal(test_case, truth_case)
  
	def test_numpy_matrix_wma(self):
		test_case = calculate_numpy_matrix_wma(pd.Series(self.input_data), m=4)
		test_case = np.array(test_case.values)
		truth_case = np.array(self.truth_case_data)
		np.testing.assert_array_equal(test_case, truth_case)
	
	def test_pandas_wma(self):
		test_case = calculate_pandas_wma(pd.Series(self.input_data), m=4)
		truth_case = pd.Series(self.truth_case_data)
		np.testing.assert_array_equal(test_case, truth_case)
  
	def test_pandas_wma_2(self):
		test_case = _calculate_pandas_wma_2(pd.Series(self.input_data), m=4)
		truth_case = pd.Series(self.truth_case_data)
		np.testing.assert_array_equal(test_case, truth_case)
  
	def test_pandas_wma_3(self):
		test_case = _calculate_pandas_wma_3(pd.Series(self.input_data), m=4)
		truth_case = pd.Series(self.truth_case_data)
		np.testing.assert_array_equal(test_case, truth_case)
  
class TestHMA(unittest.TestCase):
	def setUp(self):
		self.input_data = [1,2,3,4,5,6,7,8,9,10]
		self.truth_case_data_hma = [np.NaN, np.NaN, np.NaN, np.NaN, 
                              		5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  
	def test_pure_python_hma(self):
		test_case = np.array(_calculate_pure_python_hma(self.input_data, m=4), 
                       			dtype=np.float64)
		truth_case = np.array(self.truth_case_data_hma, dtype=np.float64)
		np.testing.assert_allclose(test_case, truth_case, rtol=1e-07)
  
	def test_numpy_hma(self):
		test_case = _calculate_numpy_hma(np.array(self.input_data), m=4)
		truth_case = np.array(self.truth_case_data_hma)
		np.testing.assert_allclose(test_case, truth_case, rtol=1e-07)
  
	def test_numpy_matrix_hma(self):
		test_case = calculate_numpy_matrix_hma(pd.Series(self.input_data), m=4)
		test_case = np.array(test_case.values)
		truth_case = np.array(self.truth_case_data_hma)
		np.testing.assert_allclose(test_case, truth_case, rtol=1e-07)
  
	def test_pandas_hma(self):
		test_case = calculate_pandas_hma(pd.Series(self.input_data), m=4)
		truth_case = pd.Series(self.truth_case_data_hma)
		np.testing.assert_allclose(test_case, truth_case, rtol=1e-07)



if __name__ == '__main__':
	
	unittest.main()