from vqti.indicators.RSI import relative_strength_index, rsi_signal_line_calculation
from pypm import metrics, signals, data_io, simulation
from typing import List, Dict, Any
import pandas as pd

# Load in data
symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)
preference = prices.apply(metrics.calculate_rolling_sharpe_ratio, axis=0)

def run_simulation(periods: int, max_active_positions: int) -> Dict[str, Any]:

    
    # Just run apply using your signal function

    #just apply the rsi calculation and see what it looks like to determine if signal line calculation is functional
    signal = prices.apply(rsi_signal_line_calculation, args=[periods], axis=0)

    #change the signal to only represent AWU, need to find out what why the program only works with lookback greater than or equal to 40

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

        'periods': periods,
        'max_active_positions': max_active_positions,
    }
print(run_simulation(16,5))

rows = list()
for periods in [5, 10, 14, 20, 40, 80, 100]:
    for max_active_positions in [5, 20]:
        print('Simulating', '...', periods, max_active_positions)
        row = run_simulation(
            periods,
            max_active_positions
        )
        rows.append(row)
df = pd.DataFrame(rows)
