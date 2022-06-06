from pypm import metrics, signals, data_io, simulation
from typing import List
import pandas as pd

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



def rsi_signal_line_calculation(close: List[float]) -> List[float]:
    calculation_list = pure_python_relative_strength_index(close)
    print(calculation_list)
    result = []
    for x in calculation_list:
        if (x < 30):
            result.append(1)
        if (x > 70):
            result.append(-1)
        else:
            result.append(0)
    return result

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

