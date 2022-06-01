from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from aroon_oscillator import aroon_python_deque, aroon_signal_line

class HistoricalData():

    def __init__(self, data_dir: Path = None):
        self.data = None

    def _read_eod_dir_csv(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path, usecols=["date","high", "low", "close"], index_col="date")
        df = df.rename(
            columns={
                "high": file_path.stem + "_high",
                "low": file_path.stem + "_low",
                "close": file_path.stem + "_close"
            }
        )
        return df

    def load_eod_dir(self, data_dir: Path) -> None:
        self.data: pd.DataFrame = pd.concat(
                [self._read_eod_dir_csv(f) for f in data_dir.iterdir() if f.suffix == ".csv"],
                axis = 1
            )
        self.data.index = pd.to_datetime(self.data.index)

    def load_prices_csv(self, csv_path: Path) -> None:
        self.data: pd.DataFrame = pd.read_csv(
            csv_path,
            parse_dates=True
        )
        self.data = self.data.loc[:,["date", "ticker", "high", "low", "close_split_adjusted"]].rename(
            columns = {"close_split_adjusted": "close"}
        )
        self.data = self.data.pivot_table(index=["date"], columns="ticker", values=["high", "low", "close"])
        self.data = self.data.sort_index(axis=1, level=1)
        self.data.columns = [f'{y}_{x}' for x, y in self.data.columns]
        self.data = self.data.fillna(method='ffill')


class SignalCalculator():

    def __init__(self, signal_params = {}):
        self.signal_params = {
            "aroon": {"p": 25, "signal_threshold": 100}
        }
        for s in signal_params:
            for p in signal_params[s]:
                self.signal_params[s][p] = signal_params[s][p]
        self.signal_functions = {
            "aroon": {"fn": self._calculate_aroon}
        } ## TODO: implement customizable signals function dict capable of accepting multiple signals functions

    def _calculate_aroon(self, stock: str, df: pd.DataFrame) -> pd.Series:
        aroon_oscillator: List = aroon_python_deque(
            df[f"{stock}_high"].tolist(),
            df[f"{stock}_low"].tolist(),
            p=self.signal_params["aroon"]["p"])
        aroon_as_series = pd.Series(
            data = aroon_oscillator,
            name = f"{stock}",
            index = df.index
        )
        aroon_signal = aroon_signal_line(
            aroon_as_series,
            signal_threshold=self.signal_params["aroon"]["signal_threshold"]
        )
        return aroon_signal

    def _signalize_stock(self, stock, df):
        return pd.concat(
            [self.signal_functions[signal]["fn"](stock, df)
             for signal in self.signal_params.keys()
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
        self.price_df.index = pd.to_datetime(self.price_df.index)
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
        self.cagr = self._calculate_cagr(price_series)
        return_series = self._calculate_return_series(price_series)
        self.volatility = self._calculate_annualized_volatility(return_series)
        return (self.cagr - benchmark_rate) / self.volatility

    def run(self, signals_df: pd.DataFrame) -> None:
        """
        Simulate performance based upon buy/sell signals in input `signals_df`.
        """
        signals_df.index = pd.to_datetime(signals_df.index)
        equity_curve = {}
        # Loop through days in price dataframe.
        dt_index = self.price_df.index
        for dt in dt_index:
            self.current_dt = dt

            # If stock is in portfolio: sell if signal is -1, or list to buy if signal is 1
            signals = signals_df.loc[dt]
            # Sell if sell signal (-1) or if stock has disappeared from exchange (nan)
            stocks_to_sell: List[str] = [s for s in signals[signals == -1].index.tolist() if s in self.portfolio] + \
                                        [s for s in signals[signals.isna()].index.tolist() if s in self.portfolio]
            # Buy if buy signal (1)
            stocks_to_buy: List[str] = [s for s in signals[signals == 1].index.tolist() if s not in self.portfolio]

            # Sell all held stocks marked for sale
            for stock in stocks_to_sell:
                self._sell_stock(stock, fraction=1)

            # Calculate how many asset slots are available, and thus how many
            free_asset_slots = self.max_assets - len(self.portfolio.keys())
            num_stocks_to_buy = min(len(stocks_to_buy), free_asset_slots)

            # Buy stocks
            if num_stocks_to_buy > 0:
                # Spend 99% of free cash equally on stocks to be bought, but don't spend more than $10,000
                available_cash_per_asset = 0.99 * self.cash / num_stocks_to_buy
                if available_cash_per_asset > 10000:
                    available_cash_per_asset = 10000
                for stock in stocks_to_buy:
                    # Loop through stocks to buy until done, or until all asset slots are filled.
                    if len(self.portfolio.keys()) <= self.max_assets:
                        self._buy_stock(stock, available_cash_per_asset)

            # Update equity curve
            equity_curve[dt] = self.cash + self._portfolio_value()
        self.equity_curve = pd.Series(equity_curve)
        self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
        self.sharpe_ratio = self._calculate_sharpe_ratio(self.equity_curve)

if __name__ == "__main__":
    eod_data_dir: Path = Path(__file__).parent.parent / "data" / "eod"

    # Load eod CSV files into custom HistoricalData class
    hd: HistoricalData = HistoricalData()
    hd.load_eod_dir(eod_data_dir)

    # Calculate signals based upon Aroon Oscillator
    signal_calc: SignalCalculator = SignalCalculator()
    signal_df: pd.DataFrame = signal_calc.calculate_signals(hd)

    # Simulate historical performance
    sim = TradingSimulator(hd)
    sim.run(signal_df)

    # Get simulation results
    print("Sharpe Ratio:", sim.sharpe_ratio)
    print("CAGR:", sim.cagr)
    print("Volatility:", sim.volatility)
    sim.equity_curve.plot(
        title = f"Equity curve based on Aroon Oscillator, Sharpe={round(sim.sharpe_ratio, 3)}",
        ylabel = "$",
        xlabel = "Date"
    )
    plt.show()

    # print("AWU:", sim._calculate_sharpe_ratio(sim.price_df.AWU))
    # print("BMG:", sim._calculate_sharpe_ratio(sim.price_df.BMG))
    # print("CUU:", sim._calculate_sharpe_ratio(sim.price_df.CUU))
