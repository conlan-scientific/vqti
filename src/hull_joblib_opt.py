from pypm import metrics, signals, data_io, simulation
from hullma_signal import hma_trend_signal, hma_zscore_signal, hma_macd_signal
from typing import List, Dict, Any
import pandas as pd
import itertools
import time
from joblib import Parallel, delayed

symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)
preference = prices.apply(metrics.calculate_rolling_sharpe_ratio, axis=0)

def run_simulation(hma_length: int, max_active_positions: int) -> Dict[str, Any]:

    # Just run apply using your signal function
    signal = prices.apply(hma_trend_signal, args=([hma_length]), axis=0)

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

        'hma_length': hma_length,
        'max_active_positions': max_active_positions,
    }
    
     
def run():
    hma_length: List = [4, 9, 16, 25, 49, 81]
    max_active_positions: List = [10, 20, 30, 40, 50]
    parameters = list(itertools.product(hma_length, max_active_positions))
    results=((Parallel(n_jobs=8)(delayed(run_simulation)(combo[0], combo[1]) for i, combo in enumerate(parameters))))            
    df = pd.DataFrame(results)
    return df
    
if __name__ == '__main__':
    start= time.time()
    results = run()
    print(results)
    end= time.time()
    print(f'Parallel took {end-start} seconds')
    # took 34.05660676956177 seconds compared to non parallelized 168.59732389450073 seconds 