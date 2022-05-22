from pathlib import Path
from typing import List
import unittest

import numpy as np
import pandas as pd

from aroon_oscillator import aroon_python_deque, aroon_signal_line

class HistoricalData():

    def __init__(self, data_dir: Path = None):
        self.data = None

    def _read_csv(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path, usecols=["high", "low"])
        df = df.rename(
            columns={
                "high": file_path.stem + "_high",
                "low": file_path.stem + "_low"
            }
        )
        return df

    def load_data(self, data_dir: Path) -> None:
        self.data: pd.DataFrame = pd.concat(
                [self._read_csv(f) for f in data_dir.iterdir() if f.suffix == ".csv"],
                axis = 1
            )


class SignalCalculator():

    def __init__(self, signals = {"aroon": {}}):
        self.signals = signals
        self.signal_functions = {
            "aroon": {"fn": self._calculate_aroon}
        }

    def _calculate_aroon(self, stock: str, df: pd.DataFrame) -> pd.Series:
        aroon_oscillator: List = aroon_python_deque(df[f"{stock}_high"].tolist(), df[f"{stock}_low"].tolist())
        aroon_as_series = pd.Series(
            data = aroon_oscillator,
            name = f"{stock}_aroon",
            index = df.index
        )
        aroon_signal = aroon_signal_line(aroon_as_series)
        return aroon_signal

    def _signalize_stock(self, stock, df):
        return pd.concat(
            [self.signal_functions[signal]["fn"](stock, df)
             for signal in self.signals.keys()
             ],
            axis = 1
        )

    def calculate_signals(self, hd: HistoricalData):
        stocks = {s.split('_')[0] for s in hd.data.columns}
        return pd.concat(
            [self._signalize_stock(stock, hd.data) for stock in stocks],
            axis=1
        )

class HistoricalDataTestCase(unittest.TestCase):

    def test_read_csv(self):
        test_file: Path = Path(__file__).parent.parent / "data" / "eod" / "AWU.csv"
        hd_df: pd.DataFrame = HistoricalData()._read_csv(test_file)
        self.assertIn("AWU_high", hd_df.columns)
        self.assertIn("AWU_low", hd_df.columns)
        self.assertEqual(len(hd_df.columns), 2)

    def test_load_data(self):
        test_dir: Path = Path(__file__).parent.parent / "data" / "eod"
        hd: HistoricalData = HistoricalData()
        hd.load_data(test_dir)
        self.assertEqual(len(hd.data.columns), 200)
        self.assertIn("AWU_high", hd.data.columns)
        self.assertIn("AWU_low", hd.data.columns)
        self.assertIn("ZZQB_high", hd.data.columns)
        self.assertIn("ZZQB_low", hd.data.columns)

class SignalCalculatorTest(unittest.TestCase):

    def test_constructor(self):
        sd = SignalCalculator()
        self.assertIsInstance(sd, SignalCalculator)
        self.assertListEqual(list(sd.signals.keys()), ["aroon"])

    def test_calculate_signals(self):
        test_dir: Path = Path(__file__).parent.parent / "data" / "eod"
        hd: HistoricalData = HistoricalData()
        hd.load_data(test_dir)
        signal_calc = SignalCalculator()
        signal_df = signal_calc.calculate_signals(hd)
        # Check that there is 1 column for each stock
        self.assertEqual(len(signal_df.columns), 100)
        # Check that all the original companies have a signal
        self.assertListEqual(
            sorted(signal_df.columns.tolist()),
            sorted([f.stem+"_aroon" for f in test_dir.iterdir()])
        )
        # Check that the only signals in the dataframe are -1, 0, and 1.
        for col in signal_df.columns:
            for val in signal_df[col].unique():
                if not np.isnan(val):
                    self.assertIn(val, [-1.0,  0.0,  1.0])

    def test_calculate_aroon(self):
        test_dir: Path = Path(__file__).parent.parent / "data" / "eod"
        hd: HistoricalData = HistoricalData()
        hd.load_data(test_dir)
        signal_calc = SignalCalculator()
        signal = signal_calc._calculate_aroon("AWU", hd.data)

        self.assertIsInstance(signal, pd.Series)
        self.assertEqual(len(signal), 2516)
        self.assertEqual(signal.name, "AWU_aroon")