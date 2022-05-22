from pathlib import Path
import unittest

import pandas as pd

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
