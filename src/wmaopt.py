from pypm import metrics, signals, data_io, simulation
from hullma_signal import hma_trend_signal, hma_zscore_signal, hma_macd_signal, wma_trend_signal
from typing import List, Dict, Any
import pandas as pd
import itertools
import time

# assert pd.Index([close,high]).isin(df.columns)
# assert columns are floats
#instead of merge, use concat. Instead of map, use apply(lambda). 
# assert df.index.is_unique
# Load in data
symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)
preference = prices.apply(metrics.calculate_rolling_sharpe_ratio, axis=0)

def run_simulation(wma_length: int, max_active_positions: int) -> Dict[str, Any]:

    # Just run apply using your signal function
    signal = prices.apply(wma_trend_signal, args=([wma_length]), axis=0)

    # Do nothing on the last day
    signal.iloc[-1] = 0

    assert signal.index.equals(preference.index)
    assert prices.columns.equals(signal.columns)
    assert signal.columns.equals(preference.columns)

    # Run the simulator
    simulator = simulation.SimpleSimulator(
        initial_cash=100000,
        max_active_positions=max_active_positions,
        percent_slippage=0.0005,
        trade_fee=1,
    )
    simulator.simulate(prices, signal, preference)

    # Print results
    # simulator.portfolio_history.print_position_summaries()
    # simulator.print_initial_parameters()
    # simulator.portfolio_history.print_summary()
    # simulator.portfolio_history.plot()
    # simulator.portfolio_history.plot_benchmark_comparison()

    portfolio_history = simulator.portfolio_history
    return {
        'percent_return': portfolio_history.percent_return,
        'spy_percent_return': portfolio_history.spy_percent_return,
        'cagr': portfolio_history.cagr,
        'volatility': portfolio_history.volatility,
        'sharpe_ratio': portfolio_history.sharpe_ratio,
        'spy_cagr': portfolio_history.spy_cagr,
        'excess_cagr': portfolio_history.excess_cagr,
        'jensens_alpha': portfolio_history.jensens_alpha,
        'dollar_max_drawdown': portfolio_history.dollar_max_drawdown,
        'percent_max_drawdown': portfolio_history.percent_max_drawdown,
        'log_max_drawdown_ratio': portfolio_history.log_max_drawdown_ratio,
        'number_of_trades': portfolio_history.number_of_trades,
        'average_active_trades': portfolio_history.average_active_trades,
        'final_equity': portfolio_history.final_equity,

        'wma_length': wma_length,
        'max_active_positions': max_active_positions,
    }
# print(run_simulation(16,5))
'''
rows = list()
for hma_length in [4, 9, 16, 25, 49, 81]:
            for max_active_positions in [5, 20]:
                #print('Simulating', hma_length, max_active_positions)
                row = run_simulation(
                    hma_length,
                    max_active_positions
                )
                rows.append(row)
df = pd.DataFrame(rows)
print(df)
'''
start= time.time()
#wma trend signal best is 30(10-20 positions) and 100 (20-50) positions. top 3 pct return  1.33, 0.98, 0.96
wma_length: List = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200]
max_active_positions: List = [10, 20, 30, 40, 50]
parameters = list(itertools.product(wma_length, max_active_positions))
results = []
for i, combo in enumerate(parameters):
        results.append(
            run_simulation(
                wma_length=combo[0],
                max_active_positions=combo[1]
            )
        )
df = pd.DataFrame(results)
print(df)
end= time.time()
print(end - start)