from pypm import metrics, signals, data_io, simulation
from typing import List
import pandas as pd
import numpy as np

# TODO: The return value of this function should be List[float]
def pure_python_relative_strength_index(close: List[float], n: int = 14) -> List[float]:
    upList = close.clip(lower = 0)
    downList = -1 * close.clip(upper = 0)
    upEwmList = upList.ewm(com = n - 1, adjust = True, min_periods = n).mean() # Units are change-in-dollars
    downEwmList = downList.ewm(com = n - 1, adjust = True, min_periods = n).mean() # Units are change-in-dollar
    
    rsf = upEwmList / downEwmList

    rsi = 100 - (100/(1 + rsf))

    assert (len(close) == len(rsi))

    return rsi

def rsi_signal_line_calculation(close: List[float], n : int = 10) -> List[float]:
    calculation_list = pure_python_relative_strength_index(close, n)
    signal = calculation_list
    crossUp = False
    crossDown = False
    for x in range(len(calculation_list)):
        if (calculation_list[x] > 80 and not crossUp):
            signal[x] = 1
            crossUp = True
        elif (calculation_list[x] < 20 and not crossDown):
            signal[x] = -1
            crossDown = True
        elif (calculation_list[x] > 20 and calculation_list[x] < 80):
            signal[x] = 0
            crossDown = False
            crossUp = False
    # signal = np.where(calculation_list > 70, 1, 0) #crossover based, whne it crosses over 70 the first time 
    # signal = np.where(calculation_list < 30, -1, signal)
    return signal


# Load in data
symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

# Just run apply using your signal function
signal = prices.apply(rsi_signal_line_calculation, axis=0)
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

