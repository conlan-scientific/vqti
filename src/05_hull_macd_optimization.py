from pypm import metrics, signals, data_io, simulation
from vqti.indicators.hma_signals import calculate_hma_macd_signal
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

def run_hma_macd_simulation(m1: int, m2:int, sig: int, max_active_positions: int) -> Dict[str, Any]:

    # Just run apply using your signal function
    signal = prices.apply(calculate_hma_macd_signal, args=([m1,m2]), axis=0)

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

        'm1': m1,
        'm2': m2,
        'sig': sig,
        'max_active_positions': max_active_positions,
    }


start= time.time()
#hma macd signal. Best 49, 81, 4-16, 10. best pct 1.067228, 0.85, 
m1: List = [16, 25, 49, 81]
m2: List = [25, 49, 81]
sig: List = [4, 9, 16]
max_active_positions: List = [10, 20, 30, 40, 50]
parameters = list(itertools.product(m1, m2, sig, max_active_positions))
results = []
for i, combo in enumerate(parameters):
    if combo[0] < combo[1]:
        if combo[2] < combo[0]:
            results.append( 
                run_hma_macd_simulation( 
                    m1=combo[0], 
                    m2=combo[1], 
                    sig=combo[2], 
                    max_active_positions=combo[3]
                )
            )
        else:
            pass    
    else:
        pass

df = pd.DataFrame(results)
print(df)
end = time.time()
print(end - start)
