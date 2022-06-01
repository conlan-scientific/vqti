from operator import rshift
from pypm import metrics, signals, data_io, simulation
from typing import List
import numpy as np
import pandas as pd


def relative_strength_index(close: List[float], n: int = 10) -> List[float]:
    close_delta = close.diff() # Units are change-in-dollars
    upList = close_delta.clip(lower = 0)
    downList = -1 * close_delta.clip(upper = 0)
    ema_up = upList.ewm(com = n - 1, adjust = True, min_periods = n).mean() # Units are change-in-dollars
    ema_down = downList.ewm(com = n - 1, adjust = True, min_periods = n).mean()# Units are change-in-dollars

    rsf = ema_up / ema_down

    rsi = 100 - (100/(1 + rsf))
    assert (len(close) == len(rsi))

    return rsi

def rsi_signal_line_calculation(close: pd.Series) -> pd.Series:
    calculation_list = relative_strength_index(close)
    signal = np.where(calculation_list > 70, 1, 0)
    signal = np.where(calculation_list < 30, -1, signal)

    return signal

# Load in data
symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

# Just run apply using your signal function
signal = prices.apply(rsi_signal_line_calculation, axis=0)
# signal *= -1

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

