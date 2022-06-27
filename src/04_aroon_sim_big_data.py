from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from aroon_simulator import HistoricalData, SignalCalculator, TradingSimulator

def read_prices_df(csv_path: Path):
    df = pd.read_csv(
        csv_path,
        parse_dates=True
    )
    df = df.loc[:, ["date", "ticker", "high", "low", "close_split_adjusted"]].rename(
        columns={"close_split_adjusted": "close"}
    )

    df = df.pivot_table(index=["date"], columns="ticker", values=["high", "low", "close"])
    df = df.sort_index(axis=1, level=1)
    df.columns = [f'{y}_{x}' for x, y in df.columns]
    return df

if __name__ == "__main__":
    prices_file = Path("~") / "Downloads" / "prices.csv"
    print(prices_file)

    hd = HistoricalData()
    hd.load_prices_csv(csv_path=prices_file)

    signal_calc: SignalCalculator = SignalCalculator()
    signal_df: pd.DataFrame = signal_calc.calculate_signals(hd)

    sim = TradingSimulator(hd)
    sim.run(signal_df)

    # Get simulation results
    print("Sharpe Ratio:", sim.sharpe_ratio)
    print("CAGR:", sim.cagr)
    print("Volatility:", sim.volatility)
    sim.equity_curve.plot(
        title=f"Equity curve based on Aroon Oscillator, Sharpe={round(sim.sharpe_ratio, 3)}",
        ylabel="$",
        xlabel="Date"
    )
    plt.show()


    # full_prices_file = Path("~") / "Downloads" / "prices.csv"
    # pdf = read_prices_df(full_prices_file)