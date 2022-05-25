from pypm import metrics, signals, data_io, simulation
from hullma_signal import hma_trend_signal
from typing import List
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Load in data
symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

# Just run apply using your signal function
signal = prices.apply(hma_trend_signal, axis=0)
signal.iloc[-1] = 0

# import hashlib
# print(hashlib.md5(signal.to_csv().encode()).hexdigest())

# signal *= -1
# print(signal)
# preference = pd.DataFrame(
#     np.random.random(prices.shape), 
#     index=prices.index, 
#     columns=prices.columns,
# )
preference = pd.DataFrame(0, index=prices.index, columns=prices.columns)

assert signal.index.equals(preference.index)
assert prices.index.equals(preference.index)
assert prices.columns.equals(signal.columns)
assert signal.columns.equals(preference.columns)

# Run the simulator
simulator = simulation.SimpleSimulator(
    initial_cash=100000,
    max_active_positions=20,
    percent_slippage=0.0000,
    trade_fee=0,
)
simulator.simulate(prices, signal, preference)


# Print results
simulator.portfolio_history.print_position_summaries()
simulator.print_initial_parameters()
simulator.portfolio_history.print_summary()
simulator.portfolio_history.plot()
simulator.portfolio_history.plot_benchmark_comparison()