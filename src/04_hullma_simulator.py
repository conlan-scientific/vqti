from pypm import metrics, signals, data_io, simulation
from vqti.indicators.hma_signals import (
    calculate_hma_trend_signal,
    calculate_hma_macd_signal,
    calculate_hma_crossover_signal,
    calculate_hma_price_crossover_signal,
    calculate_hma_zscore_signal,
    calculate_hma_zscore
)
from typing import List
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Load in data
symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

#show the distribution of the zscore. ~ -0.7 - 0.7
zscore_hist = prices.apply(calculate_hma_zscore, args=[25,45], axis=0)
plt.hist(zscore_hist)
plt.show()

# Just run apply using your signal function
# use args to implement series and m values in a different script
signal = prices.apply(calculate_hma_zscore_signal, args=[25,45], axis=0)
signal.iloc[-1] = 0
preference = prices.apply(metrics.calculate_rolling_sharpe_ratio, axis=0)

# import hashlib
# this checks to see if the there is any randomness in a function.
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
    max_active_positions=20,
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