from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
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
    This is an O(n) ? algorithm for lists of length n and lookback of p

    highs and lows are highs and lows of daily candlesticks

    Based on the accepted answer to this StackOverflow question:
    https://stackoverflow.com/questions/14823713/efficient-rolling-max-and-min-window

    Justification for why it's O(n)
    """
    period=p+1

    min_list: List = []
    min_idx_list: List = []
    periods_since_min_list: List = []
    aroon_low_list: List = []

    assert len(high) == len(low), 'Lists are unequal length.'
    low_deque = deque()
    for idx, val in enumerate(low):
        while len(low_deque) > 0 and idx >= low_deque[0][0] + period:
            low_deque.popleft()
        while len(low_deque) > 0 and low_deque[len(low_deque)-1][1] >= val:
            low_deque.pop()
        low_deque.append(
            (idx, val)
        )
        lookback_min = low_deque[0][1]
        lookback_min_idx = low_deque[0][0]
        periods_since_min = idx - lookback_min_idx
        aroon_low = 100 * (p - periods_since_min) / p
        # print(f"On position {idx} and the low min is {lookback_min} at index {lookback_min_idx}")
        # print("Current state of deque:", low_deque)
        min_list.append(lookback_min) if idx >= p else min_list.append(None)
        min_idx_list.append(lookback_min_idx) if idx >= p else min_idx_list.append(None)
        periods_since_min_list.append(periods_since_min) if idx >= p else periods_since_min_list.append(None)
        aroon_low_list.append(aroon_low) if idx >= p else aroon_low_list.append(None)
    # print(min_list)
    # print(min_idx_list)
    # print(periods_since_min_list)

    max_list: List = []
    max_idx_list: List = []
    periods_since_max_list: List = []
    aroon_high_list: List = []


    high_deque = deque()
    for idx, val in enumerate(high):
        while len(high_deque) > 0 and idx >= high_deque[0][0] + period:
            high_deque.popleft()
        while len(high_deque) > 0 and high_deque[len(high_deque)-1][1] <= val:
            high_deque.pop()
        high_deque.append(
            (idx, val)
        )
        lookback_max = high_deque[0][1]
        lookback_max_idx = high_deque[0][0]
        periods_since_max = idx - lookback_max_idx
        aroon_high = 100 * (p - periods_since_max) / p
        # print(f"On position {idx} and the low max is {lookback_max} at index {lookback_max_idx}")
        # print("Current state of deque:", high_deque)
        max_list.append(lookback_max) if idx >= p else max_list.append(None)
        max_idx_list.append(lookback_max_idx) if idx >= p else max_idx_list.append(None)
        periods_since_max_list.append(periods_since_max) if idx >= p else periods_since_max_list.append(None)
        aroon_high_list.append(aroon_high) if idx >= p else aroon_high_list.append(None)
    # print(max_list)
    # print(max_idx_list)
    # print(periods_since_max_list)

    # print(aroon_high_list)
    # print(aroon_low_list)

    aroon_oscillator = [
        high - low if high != None and low != None
        else None
        for high, low in zip(aroon_high_list, aroon_low_list)
    ]
    # print(aroon_oscillator)
    return aroon_oscillator


def test_python_deque_aroon():
    ground_truth_result = [None, -100.0, -100.0, -100.0, 0.0, -100.0, 100.0, 100.0, 100.0, 100.0]
    test_result = aroon_python_deque(
        [10,9,8,7,7,3,4,5,6,7],
        [10,9,8,7,7,3,4,5,6,7],
        p=1)
    print(test_result)
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
    result: pd.Series = aroon_pandas(df.high, df.low)
