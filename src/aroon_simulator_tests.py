from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from aroon_simulator import HistoricalData, SignalCalculator, TradingSimulator

class HistoricalDataTestCase(unittest.TestCase):

    def test_read_csv(self):
        test_file: Path = Path(__file__).parent.parent / "data" / "eod" / "AWU.csv"
        hd_df: pd.DataFrame = HistoricalData()._read_eod_dir_csv(test_file)
        self.assertIn("AWU_high", hd_df.columns)
        self.assertIn("AWU_low", hd_df.columns)
        self.assertEqual(len(hd_df.columns), 3)

    def test_load_data(self):
        test_dir: Path = Path(__file__).parent.parent / "data" / "eod"
        hd: HistoricalData = HistoricalData()
        hd.load_eod_dir(test_dir)
        self.assertEqual(len(hd.data.columns), 300)
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
        hd.load_eod_dir(test_dir)
        signal_calc = SignalCalculator()
        signal_df = signal_calc.calculate_signals(hd)
        # Check that there is 1 column for each stock
        self.assertEqual(len(signal_df.columns), 100)
        # Check that all the original companies have a signal
        self.assertListEqual(
            sorted(signal_df.columns.tolist()),
            sorted([f.stem for f in test_dir.iterdir()])
        )
        # Check that the only signals in the dataframe are -1, 0, and 1.
        for col in signal_df.columns:
            for val in signal_df[col].unique():
                if not np.isnan(val):
                    self.assertIn(val, [-1.0,  0.0,  1.0])

    def test_calculate_aroon(self):
        test_dir: Path = Path(__file__).parent.parent / "data" / "eod"
        hd: HistoricalData = HistoricalData()
        hd.load_eod_dir(test_dir)
        signal_calc = SignalCalculator()
        signal = signal_calc._calculate_aroon("AWU", hd.data)

        self.assertIsInstance(signal, pd.Series)
        self.assertEqual(len(signal), 2516)
        self.assertEqual(signal.name, "AWU")

    def test_default_aroon_lookback_window(self):
        test_dir: Path = Path(__file__).parent.parent / "data" / "eod"
        hd: HistoricalData = HistoricalData()
        hd.load_eod_dir(test_dir)
        signal_calc = SignalCalculator()
        signal = signal_calc._calculate_aroon("AWU", hd.data)
        self.assertEqual(len(signal), 2516)
        missing = signal.isna()
        self.assertTrue(missing.iloc[0])
        self.assertTrue(missing.iloc[24])
        self.assertFalse(missing.iloc[25])
        self.assertFalse(missing.iloc[2515])

    def test_small_aroon_lookback_window(self):
        test_dir: Path = Path(__file__).parent.parent / "data" / "eod"
        hd: HistoricalData = HistoricalData()
        hd.load_eod_dir(test_dir)
        signal_calc = SignalCalculator(
            signal_params={
                "aroon": {"p": 5}
            }
        )
        signal = signal_calc._calculate_aroon("AWU", hd.data)
        self.assertEqual(len(signal), 2516)
        missing = signal.isna()
        self.assertTrue(missing.iloc[0])
        self.assertTrue(missing.iloc[4])
        self.assertFalse(missing.iloc[5])
        self.assertFalse(missing.iloc[2515])

class TradingSimulatorTest(unittest.TestCase):

    def setUp(self) -> None:
        test_dir: Path = Path(__file__).parent.parent / "data" / "eod"
        self.hd: HistoricalData = HistoricalData()
        self.hd.load_eod_dir(test_dir)
        self.signal_calc = SignalCalculator()
        self.signal_df = self.signal_calc.calculate_signals(self.hd)

    def test_simulator_construction(self):
        sim = TradingSimulator(self.hd)
        self.assertIsInstance(sim, TradingSimulator)
        self.assertEqual(sim.max_assets, 20)
        self.assertEqual(sim.cash, 100000)

    # @unittest.skip
    def test_simulation_run(self):
        sim = TradingSimulator(self.hd)
        sim.run(self.signal_df)
        print(sim.sharpe_ratio)

    def test_get_share_price(self):
        sim = TradingSimulator(self.hd)
        self.assertEqual(191.7, sim._share_price("AWU", "2010-01-04"))
        sim.current_dt = sim.price_df.index[1]
        self.assertEqual(188.17, sim._share_price("AWU"))
        self.assertEqual(42.95, sim._share_price("AXC"))

    def test_sell_all_of_one_stock(self):
        sim = TradingSimulator(self.hd)
        sim.current_dt = sim.price_df.index[1]
        sim.portfolio["AWU"] = 2
        sim._sell_stock("AWU", fraction = 1)
        self.assertNotIn("AWU", sim.portfolio)
        self.assertEqual(sim.cash, 100376.34)

    def test_sell_half_of_one_stock(self):
        sim = TradingSimulator(self.hd)
        sim.current_dt = sim.price_df.index[1]
        sim.portfolio["AWU"] = 2
        sim._sell_stock("AWU", fraction = 0.5)
        self.assertEqual(sim.portfolio["AWU"], 1)
        self.assertEqual(sim.cash, 100188.17)

    def test_buy_stock(self):
        sim = TradingSimulator(self.hd)
        sim.current_dt = sim.price_df.index[1]
        sim._buy_stock("AWU", 1000)
        self.assertEqual(sim.portfolio["AWU"], 1000/188.17)
        self.assertEqual(sim.cash, 99000)

    def test_portfolio_balance_calculation(self):
        sim = TradingSimulator(self.hd)
        sim.current_dt = sim.price_df.index[1]
        sim.portfolio = {
            "AWU": 1,
            "AXC": 2
        }
        portfolio_value = sim._portfolio_value()
        self.assertEqual(188.17 + 2 * 42.95, portfolio_value)
