from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
import unittest

# HMA= WMA(2*WMA(n/2) − WMA(n)),sqrt(n))
# recommended m = 4, 9, 16, 25, 49, 81


# @time_this
def pure_python_wma(values: List[float], m: int=10)-> List[float]:
	"""
	This is O(nm).

	There is no O(n) solution to this algorithm, which is true of all filter
	operations.
	"""
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	# Building a triangular filter/weight/convolution array
	#       x
	#     x x
	#   x x x
	# x x x x
	# t ---->
	weighting = []
	for i in range(1, m+1):
		weighting.append(i)

	# Initial values
	moving_average = [None] * (m-1)
	
	# Apply it
	for i in range(m-1, len(values)):
		the_average = np.average(values[(i-m+1):i+1], weights=weighting)
		moving_average.append(the_average)

	return moving_average


# @time_this
def pure_python_hma(values: List[float], m: int=10) -> List[float]:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
 
	wma1 = np.array(pure_python_wma(values, int(m/2)))
	# multiply wma1 by 2 while keeping nan values
	wma1_multiplied = [None] * (int(m/2) -1)
	for i in range((int(m/2)-1), len(wma1)):
		y = wma1[i] *2
		wma1_multiplied.append(y)
  
	wma2 = np.array(pure_python_wma(values, m))
	# subtract wma2 from wma1 multiplied while keeping null values
	raw_hma = [0] * (m-1)
	for i in range((m-1), len(wma2)):
		raw_hma.append(np.subtract(wma1_multiplied[i], wma2[i]))
	hma= pure_python_wma(raw_hma, int(np.sqrt(m)))
	hma[0:m] = [None] * m
	return hma

# @time_this
def numpy_wma(values: np.array, m: int=10) -> np.array:
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

# @time_this
# fastest wma
def numpy_matrix_wma(values: np.ndarray, m: int) -> np.ndarray:
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

# @time_this
def numpy_hma(values: np.ndarray, m: int=10) -> np.array:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return numpy_wma((2* numpy_wma(values, int(m/2))) - (numpy_wma(values, m)), int(np.sqrt(m)))

# @time_this
# fastest hma
def numpy_matrix_hma(values: np.ndarray, m: int=10) -> np.array:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return numpy_matrix_wma((2* numpy_matrix_wma(values, int(m/2))) - (numpy_matrix_wma(values, m)), int(np.sqrt(m)))

# @time_this
# fastest pandas wma
def pandas_wma(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return values.rolling(m).apply(lambda x: ((np.arange(m)+1)*x).sum()/(np.arange(m)+1).sum(), raw=True)

# @time_this  
def pandas_wma_2(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	weights = []
	for i in range(1, m+1):
		weights.append(i)
	sum_weights = np.sum(weights)
	return values.rolling(window=m).apply(lambda x: np.sum(weights*x) / sum_weights)

# @time_this
def pandas_wma_3(values: pd.Series, m: int=10) -> pd.Series:
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

# @time_this
def pandas_hma(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return pandas_wma(2* pandas_wma(values, int(m/2)) - (pandas_wma(values, m)), int(np.sqrt(m)))

####### TESTING FUNCTIONS #######

class TestWMA(unittest.TestCase):
    
	def test_pure_python_wma(self):
		input_data = [1,2,3,4,5,6,7,8,9,10]
		truth_case_data = [None, None, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
		test_case = pure_python_wma(input_data, m=4)
		self.assertEqual(test_case, truth_case_data)
	
	def test_numpy_wma(self):
		input_data = [1,2,3,4,5,6,7,8,9,10]
		truth_case_data = [np.NaN, np.NaN, np.NaN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
		test_case = numpy_wma(np.array(input_data), m=4)
		truth_case = np.array(truth_case_data)
		np.testing.assert_array_equal(test_case, truth_case)
  
	def test_numpy_matrix_wma(self):
		input_data = [1,2,3,4,5,6,7,8,9,10]
		truth_case_data = [np.NaN, np.NaN, np.NaN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
		test_case = numpy_matrix_wma(np.array(input_data), m=4)
		truth_case = np.array(truth_case_data)
		np.testing.assert_array_equal(test_case, truth_case)
	
	def test_pandas_wma(self):
		input_data = [1,2,3,4,5,6,7,8,9,10]
		truth_case_data = [np.NaN, np.NaN, np.NaN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
		test_case = pandas_wma(pd.Series(input_data), m=4)
		truth_case = pd.Series(truth_case_data)
		np.testing.assert_array_equal(test_case, truth_case)
  
	def test_pandas_wma_2(self):
		input_data = [1,2,3,4,5,6,7,8,9,10]
		truth_case_data = [np.NaN, np.NaN, np.NaN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
		test_case = pandas_wma_2(pd.Series(input_data), m=4)
		truth_case = pd.Series(truth_case_data)
		np.testing.assert_array_equal(test_case, truth_case)
  
	def test_pandas_wma_3(self):
		input_data = [1,2,3,4,5,6,7,8,9,10]
		truth_case_data = [np.NaN, np.NaN, np.NaN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
		test_case = pandas_wma_3(pd.Series(input_data), m=4)
		truth_case = pd.Series(truth_case_data)
		np.testing.assert_array_equal(test_case, truth_case)
  
class TestHMA(unittest.TestCase):
 
	def test_pure_python_hma(self):
		input_data = [1,2,3,4,5,6,7,8,9,10]
		truth_case_data_hma = [None, None, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
		test_case = pure_python_hma(input_data, m=4)
		test_case = np.array(test_case, dtype=np.float64)
		truth_case = np.array(truth_case_data_hma, dtype=np.float64)
		np.testing.assert_allclose(test_case, truth_case, rtol=1e-07)
  
	def test_numpy_hma(self):
		input_data = [1,2,3,4,5,6,7,8,9,10]
		truth_case_data_hma = [np.NaN, np.NaN, np.NaN, np.NaN, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
		test_case = numpy_hma(np.array(input_data), m=4)
		truth_case = np.array(truth_case_data_hma)
		np.testing.assert_allclose(test_case, truth_case, rtol=1e-07)
  
	def test_numpy_matrix_hma(self):
		input_data = [1,2,3,4,5,6,7,8,9,10]
		truth_case_data_hma = [np.NaN, np.NaN, np.NaN, np.NaN, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
		test_case = numpy_matrix_hma(np.array(input_data), m=4)
		truth_case = np.array(truth_case_data_hma)
		np.testing.assert_allclose(test_case, truth_case, rtol=1e-07)
  
	def test_pandas_hma(self):
		input_data = [1,2,3,4,5,6,7,8,9,10]
		truth_case_data_hma = [np.NaN, np.NaN, np.NaN, np.NaN, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
		test_case = pandas_hma(pd.Series(input_data), m=4)
		truth_case = pd.Series(truth_case_data_hma)
		np.testing.assert_allclose(test_case, truth_case, rtol=1e-07)

if __name__ == '__main__':
	
	df = load_eod('AWU')
	# print(df)
		
	'''
	result = pure_python_wma(df.close.tolist(), 4)
	print('Done!')
	result = pure_python_hma(df.close.tolist(), 4)
	print('Done!')
	result = numpy_wma(df.close.to_numpy(), 4)
	print('Done!')
	result = numpy_matrix_wma(df.close.to_numpy(), 4)
	print('Done!')
	result = numpy_hma(df.close.to_numpy(), 4)
	print("Done!")
	result = numpy_matrix_hma(df.close.to_numpy(), 4)
	print("Done!")
	result = pandas_wma(df.close, 4)
	print("Done!")
	result = pandas_hma(df.close, 4)
	print("Done!")
	'''
	# unittest.main()
 
	# pure python wma test
	input_data = [1,2,3,4,5,6,7,8,9,10]
	truth_case_data = [None, None, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
	test_case = pure_python_wma(input_data, m=4)
	truth_case = truth_case_data
	assert test_case == truth_case, 'Test failed.'
	# numpy wma test
	input_data = [1,2,3,4,5,6,7,8,9,10]
	truth_case_data = [np.NaN, np.NaN, np.NaN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
	test_case = numpy_wma(np.array(input_data), m=4)
	truth_case = np.array(truth_case_data)
	assert np.allclose(test_case, truth_case, equal_nan=True), 'numpy_wma Test failed.'
	# numpy matrix wma test
	test_case = numpy_matrix_wma(np.array(input_data), m=4)
	truth_case = np.array(truth_case_data)
	assert np.allclose(test_case, truth_case, equal_nan=True), 'numpy_wma Test failed.'
 
	#pandas wma test
	test_case = pandas_wma(pd.Series(input_data), m=4)
	truth_case = pd.Series(truth_case_data)
	assert np.allclose(test_case, truth_case, equal_nan=True), 'pandas_wma Test failed.'
	# pandas wma 2 test
	test_case = pandas_wma_2(pd.Series(input_data), m=4)
	truth_case = pd.Series(truth_case_data)
	assert np.allclose(test_case, truth_case, equal_nan=True), 'pandas_wma_2 Test failed.'
	# pandas wma 3 test
	test_case = pandas_wma_3(pd.Series(input_data), m=4)
	truth_case = pd.Series(truth_case_data)
	assert np.allclose(test_case, truth_case, equal_nan=True), 'pandas_wma_3 Test failed.'
	
	# pure python hma test
	input_data_hma = [1,2,3,4,5,6,7,8,9,10]
	truth_case_data_hma = [None, None, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	test_case = pure_python_hma(input_data_hma, m=4)
	test_case = np.array(test_case, dtype=np.float64)
	truth_case = np.array(truth_case_data_hma, dtype=np.float64)
	assert np.testing.assert_allclose(test_case, truth_case, rtol=1e-07), 'pure_python_hma Test failed.'
  
	# numpy hma test
	input_data = [1,2,3,4,5,6,7,8,9,10]
	truth_case_data_hma = [np.NaN, np.NaN, np.NaN, np.NaN, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	test_case = numpy_hma(np.array(input_data_hma), m=4)
	truth_case = np.array(truth_case_data_hma)
	assert np.allclose(test_case, truth_case, equal_nan=True), 'numpy_hma Test failed.'
  
	# numpy matrix hma test
	input_data = [1,2,3,4,5,6,7,8,9,10]
	truth_case_data_hma = [np.NaN, np.NaN, np.NaN, np.NaN, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	test_case = numpy_matrix_hma(np.array(input_data_hma), m=4)
	truth_case = np.array(truth_case_data_hma)
	assert np.allclose(test_case, truth_case, equal_nan=True), 'numpy_hma Test failed.'

	#pandas hma test
	test_case = pandas_hma(pd.Series(input_data_hma), m=4)
	truth_case = pd.Series(truth_case_data_hma)
	assert np.allclose(test_case, truth_case, equal_nan=True), 'pandas_hma Test failed.'
 