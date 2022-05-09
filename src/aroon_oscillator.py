from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List


@time_this
def aroon_python_basic(high: List[float], low: List[float], p: int=25) -> List[float]:
    """
    This is an O(?) algorithm
    """
    result: List = [None] * (p-1)
    for i in range(p, len(low)+1):
        low_window_min: float = min(low[i-p:i])
        high_window_max: float = max(high[i-p:i])
        aroon_down: float = 100*(p-low[i-p:i][::-1].index(low_window_min))/p
        aroon_up: float = 100*(p-high[i-p:i][::-1].index(high_window_max))/p
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
    This is an O(?) algorithm
    """

    high_idx: pd.Series = high.reset_index().index - \
               high.reset_index().high.rolling(window=p).apply(lambda x: pd.Series(x).idxmax())
    low_idx: pd.Series = low.reset_index().index - \
              low.reset_index().low.rolling(window=p).apply(lambda x: pd.Series(x).idxmax())
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
    result: pd.Series = aroon_pandas(df.high, df.low, p=2)
