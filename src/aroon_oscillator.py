from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Deque
from collections import deque


@time_this
def aroon_python_basic(high: List[float], low: List[float], p: int=25) -> List[float]:
    """
    This is an O(np) algorithm a lists of length n and lookback of p

    highs and lows are highs and lows of daily candlesticks
    """
    result: List = [None] * (p)

    assert len(high) == len(low), 'Lists are unequal length.'

    # TODO: Compute the rolling min and max as lists up here
    # ...
    # TODO: Compute aroon_down_idx in a loop in advance (maybe?)

    # loop is approx n elements
    for i in range(p, len(low)):
        # Each of these is a loop of p elements
        # Rolling min and max of length p
        low_window_min: float = min(low[i-p:i+1])
        high_window_max: float = max(high[i-p:i+1])

        # Each of these is also O(p)
        # Find the rolling argmin of a length p lookback window on low
        aroon_down_idx: int = low[i-p:i+1][::-1].index(low_window_min)
        aroon_down: float = 100*(p-aroon_down_idx)/p

        # Find the rolling argmax of a length p lookback window on high
        aroon_up_idx: int = high[i-p:i+1][::-1].index(high_window_max)
        aroon_up: float = 100*(p-aroon_up_idx)/p

        # This is O(1) operation. Nothing to worry about here.
        aroon_oscillator: float = aroon_up - aroon_down

        result.append(aroon_oscillator)

    return result


def test_pure_python_aroon():
    ground_truth_result = [None, -100.0, -100.0, -100.0, 0.0, -100.0, 100.0, 100.0, 100.0, 100.0]
    test_result = aroon_python_basic(
        [10, 9, 8, 7, 7, 3, 4, 5, 6, 7],
        [10, 9, 8, 7, 7, 3, 4, 5, 6, 7],
        p=1)
    print(test_result)
    assert len(ground_truth_result) == len(test_result)
    for i in range(len(ground_truth_result)):
        assert ground_truth_result[i] == test_result[i]


@time_this
def aroon_pandas(high: pd.Series, low: pd.Series, p: int=25) -> pd.Series:
    """
    This is an O(np) algorithm
    """
    assert high.index.equals(low.index), 'Indexes are unequal.'

    range_index = pd.Index(list(range(high.shape[0])))

    high_idx: pd.Series = range_index - \
            high.reset_index().high.rolling(window=p).apply(lambda x: x.idxmax())

    low_idx: pd.Series = range_index - \
              low.reset_index().low.rolling(window=p).apply(lambda x: x.idxmax())

    aroon_high: pd.Series = 100 * (p - high_idx)/p
    aroon_low: pd.Series = 100 * (p - low_idx)/p
    aroon_oscillator: pd.Series = aroon_high - aroon_low
    return aroon_oscillator


def test_pandas_aroon():
    # TODO: needs date index
    # TODO: needs better manual verification
    ground_truth_result = pd.Series([None, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    test_result = aroon_pandas(
        pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        p=2)
    print(test_result)
    assert len(ground_truth_result) == len(test_result)
    for i in range(len(ground_truth_result)):
        assert ground_truth_result[i] == test_result[i]

@time_this
def aroon_python_deque(high: List[float], low: List[float], p: int=25) -> List[float]:
    """
    This is an O(n) algorithm for lists of length n and lookback of p

    highs and lows are highs and lows of daily candlesticks

    Based on the accepted answer to this StackOverflow question:
    https://stackoverflow.com/questions/14823713/efficient-rolling-max-and-min-window

    Justification for why it's O(n):
    https://stackoverflow.com/questions/53094476/why-is-the-deque-solution-to-the-sliding-window-maximum-problem-on-instead-o
    """
    assert len(high) == len(low), 'Lists are unequal length.'
    # Full lookback includes p historical periods plus the current period.
    period: int = p+1

    low_deque: Deque = deque()
    high_deque: Deque = deque()
    aroon_oscillator: List = []

    for idx, low_val in enumerate(low):
        # calculate Aroon Low
        # remove any old values from the front of deque that are not in window:
        while len(low_deque) > 0 and idx >= low_deque[0][0] + period:
            low_deque.popleft()
        # remove any values from the end of deque that are larger than current low:
        while len(low_deque) > 0 and low_deque[len(low_deque)-1][1] >= low_val:
            low_deque.pop()
        low_deque.append( (idx, low_val) ) # add this value
        lookback_min_idx: int = low_deque[0][0] # get window minimum's index from front of deque
        periods_since_min: int = idx - lookback_min_idx
        aroon_low: float = 100 * (p - periods_since_min) / p

        # calculate Aroon High. Similar procedure as Aroon Low
        high_val: float = high[idx]
        while len(high_deque) > 0 and idx >= high_deque[0][0] + period:
            high_deque.popleft()
        while len(high_deque) > 0 and high_deque[len(high_deque)-1][1] <= high_val:
            high_deque.pop()
        high_deque.append( (idx, high_val) )
        lookback_max_idx: int = high_deque[0][0]
        periods_since_max: int = idx - lookback_max_idx
        aroon_high: float = 100 * (p - periods_since_max) / p

        aroon_diff: float = aroon_high - aroon_low
        aroon_oscillator.append(aroon_diff) if idx >= p else aroon_oscillator.append(None)

    return aroon_oscillator


def test_python_deque_aroon():
    # Test Case 1
    print("Test Case 1")
    ground_truth_result = [None, -100.0, -100.0, -100.0, 0.0, -100.0, 100.0, 100.0, 100.0, 100.0]
    test_result = aroon_python_deque(
        [10,9,8,7,7,3,4,5,6,7],
        [10,9,8,7,7,3,4,5,6,7],
        p=1)
    assert len(ground_truth_result) == len(test_result)
    for i in range(len(ground_truth_result)):
        assert ground_truth_result[i] == test_result[i]

    # Test Case 2
    print("Test Case 2")
    ground_truth_result = [None, None, None, None, -100.0, -100.0, -75.0, -25.0, -25.0, 100]
    test_result = aroon_python_deque(
        [10,9,8,7,7,3,4,5,6,7],
        [10,9,8,7,7,3,4,5,6,7],
        p=4)
    assert len(ground_truth_result) == len(test_result)
    for i in range(len(ground_truth_result)):
        assert ground_truth_result[i] == test_result[i]

    # Test Case 3
    print("Test Case 3")
    ground_truth_result = [None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_result = aroon_python_deque(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        p=1)
    assert len(ground_truth_result) == len(test_result)
    for i in range(len(ground_truth_result)):
        assert ground_truth_result[i] == test_result[i]

    # Test Case 4
    print("Test Case 4")
    ground_truth_result = [None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_result = aroon_python_deque(
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        p=1)
    assert len(ground_truth_result) == len(test_result)
    for i in range(len(ground_truth_result)):
        assert ground_truth_result[i] == test_result[i]

    # Test Case 5
    print("Test Case 5")
    ground_truth_result = [None, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    test_result = aroon_python_deque(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        p=1)
    assert len(ground_truth_result) == len(test_result)
    for i in range(len(ground_truth_result)):
        assert ground_truth_result[i] == test_result[i]

    # Test Case 6
    print("Test Case 6")
    ground_truth_result = [None, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0]
    test_result = aroon_python_deque(
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        p=1)
    assert len(ground_truth_result) == len(test_result)
    for i in range(len(ground_truth_result)):
        assert ground_truth_result[i] == test_result[i]


if __name__ == '__main__':
    df = load_eod('AWU')
    # print(df)

    # df.high.plot()
    # df.low.plot()
    # plt.show()

    # test_pure_python_aroon()
    # test_python_deque_aroon()


    result: List = aroon_python_basic(df.high.tolist(), df.low.tolist())
    result: List = aroon_python_deque(df.high.tolist(), df.low.tolist())

    # test_pandas_aroon()
    #result: pd.Series = aroon_pandas(df.high, df.low)
