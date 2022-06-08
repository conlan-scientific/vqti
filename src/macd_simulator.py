from pypm import metrics, signals, data_io, simulation
from macd_signals import calc_pandas_macd_signal, calc_pandas_macd, calc_pandas_macd_signal_v2
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)
signal = prices.apply(calc_pandas_macd_signal, axis=0, result_type='expand')
signal.iloc[-1] = 0
preference = prices.apply(metrics.calculate_rolling_sharpe_ratio, axis=0)

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