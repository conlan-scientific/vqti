from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List


@time_this
def aroon_python_basic(high: List[float], low: List[float], p: int=25) -> List[float]:
    """
    This is an O(np) algorithm a lists of length n and lookback of p

    highs and lows are highs and lows of daily candlesticks
    """
    result: List = [None] * (p-1)

    assert len(high) == len(low), 'Lists are unequal length.'

    # TODO: Compute the rolling min and max as lists up here
    # ...
    # TODO: Compute aroon_down_idx in a loop in advance (maybe?)

    # loop is approx n elements
    for i in range(p, len(low)+1):

        # Each of these is a loop of p elements
        # Rolling min and max of length p
        low_window_min: float = min(low[i-p:i])
        high_window_max: float = max(high[i-p:i])

        # Each of these is also O(p)
        # Find the rolling argmin of a length p lookback window on low
        aroon_down_idx: int = low[i-p:i][::-1].index(low_window_min)
        aroon_down: float = 100*(p-aroon_down_idx)/p

        # Find the rolling argmax of a length p lookback window on high
        aroon_up_idx: int = high[i-p:i][::-1].index(high_window_max)
        aroon_up: float = 100*(p-aroon_up_idx)/p

        # This is O(1) operation. Nothing to worry about here.
        aroon_oscillator: float = aroon_up - aroon_down
        result.append(aroon_oscillator)

    return result


def test_pure_python_aroon():
    # TODO: needs better manual verification
    ground_truth_result = [None, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    test_result = aroon_python_basic([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p=2)
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


if __name__ == '__main__':
    df = load_eod('AWU')
    # print(df)

    # df.high.plot()
    # df.low.plot()
    # plt.show()

    # test_pure_python_aroon()
    result: List = aroon_python_basic(df.high.tolist(), df.low.tolist())
    # test_pandas_aroon()
    result: pd.Series = aroon_pandas(df.high, df.low)
