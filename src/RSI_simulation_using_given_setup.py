from pypm import metrics, signals, data_io, simulation
from typing import List
import pandas as pd

def up_down_factors(close: List[float]):
    upList = []
    downList = []
    upList.append(close[0])
    downList.append(close[0])
    for x in range(1, len(close)):
        if (close[x] - close[x-1]) > 0:
            upList.append(close[x] - close[x-1]) # This is a first order calc
            downList.append(0)
        else:
            upList.append(0)
            downList.append(close[x-1] - close[x]) # This is a first order calc
    return upList, downList


def pure_python_smma(close: List[float], n: int = 10) -> List[float]:
    prevSmma = []
    initialSmma = sum(close, n)/n
    prevSmma.append(initialSmma)

    for x in range(1, n):
        prevSum = prevSmma[x - 1] * n
        newSmma = (prevSum - prevSmma[x - 1] + close[x])/n
        prevSmma.append(newSmma)
    
    return prevSmma

def relative_strength_index_calculation(rsf: int) -> int:
    # This forces the range into 0 to 100
    return (100 - (100/(1 + rsf)))

# TODO: The return value of this function should be List[float]
def pure_python_relative_strength_index(close: List[float], n: int = 10) -> List[float]:
    upList, downList = up_down_factors(close) # Units are change-in-dollars
    upSmmaList = pure_python_smma(upList, n) # Units are change-in-dollars
    downSmmaList = pure_python_smma(downList, n) # Units are change-in-dollars

    # TODO: The relative strength factor should be a List[float] of equal length
    # to the input. It is the element-wise ratio of SMMA-up and SMMA-down.
    result = []
    for x in range(len(close)):
        # This is a ratio, range is 0 to infinity
        relative_strength_factor = upSmmaList[x] / downSmmaList[x]
        current_RSI = relative_strength_index_calculation(relative_strength_factor)
        result.append(current_RSI)
    
    assert (len(close) == len(result))

    return result



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

