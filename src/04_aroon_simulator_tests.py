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
        self.assertListEqual(list(sd.signal_params.keys()), ["aroon"])

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

class FullTradingSimulatorTest(unittest.TestCase):

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

# class SimplifiedTradingSimulatorTest(unittest.TestCase):
#
#     def setUp(self) -> None:
#         self.hd: HistoricalData = HistoricalData()
#         self.hd.data = pd.DataFrame(
#             {
#                 "AAA_close": [1,2,3,4,5,6,7,8,9,10],
#                 "AAA_high":  [1,2,3,4,5,6,7,8,9,10],
#                 "AAA_low":   [1,2,3,4,5,6,7,8,9,10],
#                 "BBB_close": [1,2,3,4,5,5,4,3,2,1],
#                 "BBB_high":  [1,2,3,4,5,5,4,3,2,1],
#                 "BBB_low":   [1,2,3,4,5,5,4,3,2,1],
#                 "CCC_close": [10,9,8,7,6,6,7,8,9,10],
#                 "CCC_high":  [10,9,8,7,6,6,7,8,9,10],
#                 "CCC_low":   [10,9,8,7,6,6,7,8,9,10],
#                 "DDD_close": [1,2,3,4,5,6,np.nan,np.nan,np.nan,np.nan],
#                 "DDD_high":  [1,2,3,4,5,6,np.nan,np.nan,np.nan,np.nan],
#                 "DDD_low":   [1,2,3,4,5,6,np.nan,np.nan,np.nan,np.nan],
#                 "EEE_close": [np.nan,np.nan,np.nan,np.nan,1,2,3,4,5,6],
#                 "EEE_high":  [np.nan,np.nan,np.nan,np.nan,1,2,3,4,5,6],
#                 "EEE_low":   [np.nan,np.nan,np.nan,np.nan,1,2,3,4,5,6]
#             },
#             index = pd.to_datetime([
#                 "2022-01-03",
#                 "2022-01-04",
#                 "2022-01-05",
#                 "2022-01-06",
#                 "2022-01-07",
#                 "2022-01-10",
#                 "2022-01-11",
#                 "2022-01-12",
#                 "2022-01-13",
#                 "2022-01-14"
#             ])
#         )
#         self.signal_calc = SignalCalculator(
#             signal_params={
#                 "aroon": {"p": 2}
#             }
#         )
#         self.signal_df = self.signal_calc.calculate_signals(self.hd)
#
#     def test_rebalance_portfolio(self):
#         sim = TradingSimulator(self.hd, max_assets = 3)
#         sim.portfolio = {
#             "AAA": 3,
#             "BBB": 3
#         }
#         sim.cash=0.6
#         sim.current_dt = sim.price_df.index[0]
#         print(sim.current_dt)
#         print(self.signal_df)
#         sim._calculate_rebalanced_portfolio(stocks_to_buy = 1)

if __name__ == "__main__":
    unittest.main()
