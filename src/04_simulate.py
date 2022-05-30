from pypm import metrics, signals, data_io, simulation
from typing import List
import pandas as pd

# Load in data
symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

# Just run apply using your signal function
signal = prices.apply(signals.create_bollinger_band_signal, args=(40, 1.0), axis=0)
# signal *= -1

signal.iloc[-1] = 0
preference = pd.DataFrame(0, index=prices.index, columns=prices.columns)


assert signal.index.equals(preference.index)
assert prices.columns.equals(signal.columns)
assert signal.columns.equals(preference.columns)

# Run the simulator
simulator = simulation.SimpleSimulator(
    initial_cash=10000,
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

