from pypm import metrics, signals, data_io, simulation
from hullma_signal import hma_trend_signal, hma_MACD
from typing import List
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Load in data
symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

# Just run apply using your signal function
# use args to implement series and m values in a different script
signal = prices.apply(hma_trend_signal, axis=0)
signal.iloc[-1] = 0
preference = prices.apply(metrics.calculate_rolling_sharpe_ratio, axis=0)

# import hashlib
# print(hashlib.md5(signal.to_csv().encode()).hexdigest())

# make consistent preference by making it rolling sharp.
# preference is just used if you have too many buy signals, how it says which ones to buy 
# preference = pd.DataFrame(0,index=prices.index, columns=prices.columns)

assert signal.index.equals(preference.index)
assert prices.index.equals(preference.index)
assert prices.columns.equals(signal.columns)
assert signal.columns.equals(preference.columns)

# Run the simulator
simulator = simulation.SimpleSimulator(
    initial_cash=100000,
    max_active_positions=5,
    percent_slippage=0.0005,
    trade_fee=1,
)
simulator.simulate(prices, signal, preference)


# Print results
simulator.portfolio_history.print_position_summaries()
simulator.print_initial_parameters()
simulator.portfolio_history.print_summary()
simulator.portfolio_history.plot()
simulator.portfolio_history.plot_benchmark_comparison()