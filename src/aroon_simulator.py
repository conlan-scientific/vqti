from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from aroon_oscillator import aroon_python_deque, aroon_signal_line

class HistoricalData():

    def __init__(self, data_dir: Path = None):
        self.data = None

    def _read_csv(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path, usecols=["date","high", "low", "close"], index_col="date")
        df = df.rename(
            columns={
                "high": file_path.stem + "_high",
                "low": file_path.stem + "_low",
                "close": file_path.stem + "_close"
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
            name = f"{stock}",
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

class TradingSimulator():

    def __init__(self, hd, max_assets: int= 20, starting_cash: int = 100000):
        self.max_assets = max_assets
        self.current_assets = 0
        self.cash = starting_cash
        self.price_df = self._get_price_df(hd)
        self.portfolio = {}

    def _get_price_df(self, hd: HistoricalData):
        price_cols = {old: old.split("_")[0] for old in hd.data.columns if old.endswith("close")}
        price_df = hd.data.rename(columns = price_cols)[list(price_cols.values())]
        return price_df

    def _share_price(self, stock, dt = None):
        if dt == None:
            return self.price_df.loc[self.current_dt,stock]
        else:
            return self.price_df.loc[dt,stock]

    def _sell_stock(self, stock, value = None, fraction = None):
        if fraction != None:
            share_price = self._share_price(stock)
            num_shares = self.portfolio[stock]
            self.cash += fraction * num_shares * share_price
            self.portfolio[stock] -= fraction * num_shares
            if fraction == 1:
                self.portfolio.pop(stock)
        # elif value != None:

    def _buy_stock(self, stock, value):
        share_price = self._share_price(stock)
        shares_to_buy = value / share_price
        self.cash -= value
        if stock in self.portfolio:
            self.portfolio[stock] += shares_to_buy
        else:
            self.portfolio[stock] = shares_to_buy

    def _portfolio_value(self):
        asset_holdings = [
            self._share_price(stock) * self.portfolio[stock] for stock in self.portfolio
        ]
        return sum(asset_holdings)

    def _calculate_return_series(self, series: pd.Series) -> pd.Series:
        """
        From Algorithmic Trading with Python
        """
        shifted_series = series.shift(1, axis=0)
        return series / shifted_series - 1

    def _get_years_passed(self, series: pd.Series) -> float:
        """
        From Algorithmic Trading with Python p.24
        """

        start_date = series.index[0]
        end_date = series.index[-1]
        return (end_date - start_date).days / 365.25

    def _calculate_cagr(self, return_series: pd.Series) -> float:
        """
        From Algorithmic Trading with Python
        """
        value_factor = return_series.iloc[-1] / return_series.iloc[0]
        years_passed = self._get_years_passed(return_series)
        return (value_factor ** (1 / years_passed)) - 1

    def _calculate_annualized_volatility(self, series: pd.Series) -> float:
        """
        From Algorithmic Trading with Python p.24
        """
        years_passed = self._get_years_passed(series)
        entries_per_year = series.shape[0] / years_passed
        return series.std() * np.sqrt(entries_per_year)

    def _calculate_sharpe_ratio(self, price_series: pd.Series, benchmark_rate: float = 0) -> float:
        """
        From Algorithmic Trading with Python p.27
        """
        cagr = self._calculate_cagr(price_series)
        return_series = self._calculate_return_series(price_series)
        volatility = self._calculate_annualized_volatility(return_series)
        return (cagr - benchmark_rate) / volatility

    def run(self, signals_df: pd.DataFrame) -> None:
        equity_curve = {}
        dt_index = self.price_df.index
        counter = 0
        for dt in dt_index:
            self.current_dt = dt
            # print(dt)
            signals = signals_df.loc[dt]
            # print(signals)
            stocks_to_sell: List[str] = [s for s in signals[signals == -1].index.tolist() if s in self.portfolio]
            stocks_to_buy: List[str] = [s for s in signals[signals == 1].index.tolist() if s not in self.portfolio]
            for stock in stocks_to_sell:
                self._sell_stock(stock, fraction=1)
            free_asset_slots = self.max_assets - len(self.portfolio.keys())
            num_stocks_to_buy = min(len(stocks_to_buy), free_asset_slots)
            if num_stocks_to_buy > 0:
                available_cash_per_asset = 0.99 * self.cash / num_stocks_to_buy
                if available_cash_per_asset > 10000:
                    available_cash_per_asset = 10000
                for stock in stocks_to_buy:
                    if len(self.portfolio.keys()) <= self.max_assets:
                        self._buy_stock(stock, available_cash_per_asset)
            equity_curve[dt] = self.cash + self._portfolio_value()

            # print(stocks_im_going_to_buy, stocks_im_going_to_sell)
            # stocks_im_holding: List[str]  # This is determined by which stocks you bought previously
            # counter +=1
            # if counter > 23:
            #     pass
            # if counter > 28:
            #     print("Stop date", dt)
            #     print("Stop signals:", sorted(signals))
            #     print("To sell:", stocks_to_sell)
            #     print("To buy:", stocks_to_buy)
            #     print(self.portfolio)
            #     break
        self.equity_curve = pd.Series(equity_curve)
        self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
        self.sharpe_ratio = self._calculate_sharpe_ratio(self.equity_curve)

if __name__ == "__main__":
    eod_data_dir: Path = Path(__file__).parent.parent / "data" / "eod"

    hd: HistoricalData = HistoricalData()
    hd.load_data(eod_data_dir)

    signal_calc: SignalCalculator = SignalCalculator()
    signal_df: pd.DataFrame = signal_calc.calculate_signals(hd)

    sim = TradingSimulator(hd)
    sim.run(signal_df)
    print("Sharpe Ratio", sim.sharpe_ratio)
    sim.equity_curve.plot()

    plt.show()
