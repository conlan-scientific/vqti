from pypm import metrics, signals, data_io, simulation
from typing import List, Dict, Any
import pandas as pd

# Load in data
symbols: List[str] = data_io.get_all_symbols()
prices: pd.DataFrame = data_io.load_eod_matrix(symbols)
preference = pd.DataFrame(0, index=prices.index, columns=prices.columns)

def run_simulation(bollinger_band_length: int, sigma_factor: int, 
    strategy_type: str) -> Dict[str, Any]:

    # Just run apply using your signal function
    signal = prices.apply(
        signals.create_bollinger_band_signal, 
        args=(
            bollinger_band_length, 
            sigma_factor,
        ), 
        axis=0,
    )

    assert strategy_type in ('momentum', 'reversal')
    if strategy_type == 'momentum':
        signal *= -1

    # Do nothing on the last day
    signal.iloc[-1] = 0

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

        'bollinger_band_length': bollinger_band_length,
        'sigma_factor': sigma_factor,
        'strategy_type': strategy_type,
    }


rows = list()
for bollinger_band_length in [5, 10, 20, 40, 80]:
    for sigma_factor in [1, 1.5, 2, 2.5, 3, 3.5]:
        for strategy_type in ['momentum', 'reversal']:
            for ...
                for ...
                    for ...
            print('Simulating', '...', bollinger_band_length, sigma_factor, strategy_type)
            row = run_simulation(
                bollinger_band_length,
                sigma_factor,
                strategy_type,
            )
            rows.append(row)
df = pd.DataFrame(rows)










































df = Optimizer(
    bollinger_band_length=[5, 10, 20, 40, 80],
    sigma_factor=[1, 1.5, 2, 2.5, 3, 3.5],
    strategy_type=['momentum', 'reversal'],
    max_assets=[5, 10, 20,],
    simulator=run_simulation,
)














