import pandas as pd
from pypm import metrics, signals, data_io, simulation, optimization
from pypm.optimization import GridSearchOptimizer
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from itertools import product
from timeit import default_timer
from typing import Dict, Tuple, List, Callable, Iterable, Any, NewType, Mapping
from vqti.indicators.hma_signals import calculate_hma_trend_signal
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 
from IPython import embed as ipython_embed

# Performance data and parameter inputs are dictionaries
Parameters = NewType('Parameters', Dict[str, float])
Performance = simulation.PortfolioHistory.PerformancePayload # Dict[str, float]

# Simulation function must take parameters as keyword arguments pointing to 
# iterables and return a performance metric dictionary
SimKwargs = NewType('Kwargs', Mapping[str, Iterable[Any]])
SimFunction = NewType('SimFunction', Callable[[SimKwargs], Performance])

class OptimizationResult(object):
    """Simple container class for optimization data"""

    def __init__(self, parameters: Parameters, performance: Performance):

        # Make sure no collisions between performance metrics and params
        assert len(parameters.keys() & performance.keys()) == 0, \
            'parameter name matches performance metric name'

        self.parameters = parameters
        self.performance = performance

    @property
    def as_dict(self) -> Dict[str, float]:
        """Combines the dictionaries after we are sure of no collisions"""
        return {**self.parameters, **self.performance}
    

class GridSearchOptimizer(object):
    """
    A generic grid search optimizer that requires only a simulation function and
    a series of parameter ranges. Provides timing, summary, and plotting 
    utilities with return data.
    """

    def __init__(self, simulation_function: SimFunction):

        self.simulate = simulation_function
        self._results_list: List[OptimizationResult] = list()
        self._results_df = pd.DataFrame()

        self._optimization_finished = False

    def add_results(self, parameters: Parameters, performance: Performance):
        _results = OptimizationResult(parameters, performance)
        self._results_list.append(_results)

    def optimize(self, **optimization_ranges: SimKwargs):

        assert optimization_ranges, 'Must provide non-empty parameters.'    
        # Convert all iterables to lists
        param_ranges = {k: list(v) for k, v in optimization_ranges.items()}
        self.param_names = param_names = list(param_ranges.keys())

        # Count total simulation
        n = total_simulations = np.prod([len(r) for r in param_ranges.values()])

        total_time_elapsed = 0   
        print(f'Starting simulation ...')
        timer_start = default_timer()
        print(f'Simulating 1 / {n} ...', end='\r')
        # results = returns a list of dictionaries
        #[{'percent_return': -0.3434220047247627, 'spy_percent_return': 1.8325266214908038, 
        # 'cagr': -0.041248417793209535, ......... , 'final_equity': 6565.779952752373}, {'percent_return': -0.275879124922628, 
        # 'spy_percent_return': 1.8325266214908038, 'cagr': -0.03180281922600281, 'volatility': 0.15188092696177793}]
        results = Parallel(n_jobs=8,verbose=50)(delayed(self.simulate)(params[0],params[1]) for i, params in enumerate(product(*param_ranges.values())))
        # parameters = 
        #{'hma_trend_n': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        #    'sharpe_n': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        parameters = {n: param for n, param in zip(param_names, param_ranges.values())}
        ipython_embed()
        #for i, params in 
        self.add_results(parameters, results)
        timer_end = default_timer()
        total_time_elapsed += timer_end - timer_start 

        print(f'Simulated {total_simulations} / {total_simulations} ...')
        print(f'Elapsed time: {total_time_elapsed:.0f}s')
        print(f'Done.')

        self._optimization_finished = True

    def _assert_finished(self):
        assert self._optimization_finished, \
            'Run self.optimize before accessing this method.'

    @property
    def results(self) -> pd.DataFrame:
        self._assert_finished()
        if self._results_df.empty:

            _results_list = self._results_list
            self._results_df = pd.DataFrame([r.as_dict for r in _results_list])

            _columns = set(list(self._results_df.columns.values))
            _params = set(self.param_names)
            self.metric_names = list(_columns - _params)

        return self._results_df

    def print_summary(self):
        df = self.results
        metric_names = self.metric_names

        print('Summary statistics')
        print(df[metric_names].describe().T)

    def get_best(self, metric_name: str) -> pd.DataFrame:
        """
        Sort the results by a specific performance metric
        """
        self._assert_finished()

        results = self.results
        param_names = self.param_names
        metric_names = self.metric_names

        assert metric_name in metric_names, 'Not a performance metric'
        partial_df = self.results[param_names+[metric_name]]

        return partial_df.sort_values(metric_name, ascending=False)

    def plot_1d_hist(self, x, show=True):
        self.results.hist(x)
        if show:
            plt.show()

    def plot_2d_line(self, x, y, show=True, **filter_kwargs):
        _results = self.results
        for k, v in filter_kwargs.items():
            _results = _results[getattr(_results, k) == v]

        ax = _results.plot(x, y)
        if filter_kwargs:
            k_str = ', '.join([f'{k}={v}' for k,v in filter_kwargs.items()])
            ax.legend([f'{x} ({k_str})'])

        if show:
            plt.show()

    def plot_2d_violin(self, x, y, show=True):
        """
        Group y along x then plot violin charts
        """
        x_values = self.results[x].unique()
        x_values.sort()

        y_by_x = OrderedDict([(v, []) for v in x_values])
        for _, row in self.results.iterrows():
            y_by_x[row[x]].append(row[y])

        fig, ax = plt.subplots()

        ax.violinplot(dataset=list(y_by_x.values()), showmedians=True)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_xticks(range(0, len(y_by_x)+1))
        ax.set_xticklabels([''] + list(y_by_x.keys()))
        if show:
            plt.show()

    def plot_3d_mesh(self, x, y, z, show=True, **filter_kwargs):
        """
        Plot interactive 3d mesh. z axis should typically be performance metric
        """
        _results = self.results
        fig = plt.figure()
        ax = Axes3D(fig)

        for k, v in filter_kwargs.items():
            _results = _results[getattr(_results, k) == v]

        X, Y, Z = [getattr(_results, attr) for attr in (x, y, z)]
        ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        if show:
            plt.show()

    def plot(self, *attrs: Tuple[str], show=True, 
        **filter_kwargs: Dict[str, Any]):
        """
        Attempt to intelligently dispatch plotting functions based on the number
        and type of attributes. Last argument should typically be the 
        performance metric.
        """
        self._assert_finished()
        param_names = self.param_names
        metric_names = self.metric_names

        if len(attrs) == 3:
            assert attrs[0] in param_names and attrs[1] in param_names, \
                'First two positional arguments must be parameter names.'

            assert attrs[2] in metric_names, \
                'Last positional argument must be a metric name.'

            assert len(filter_kwargs) + 2 == len(param_names), \
                'Must filter remaining parameters. e.g. p_three=some_number.'

            self.plot_3d_mesh(*attrs, show=show, **filter_kwargs)

        elif len(attrs) == 2:
            if len(param_names) == 1 or filter_kwargs:
                self.plot_2d_line(*attrs, show=show, **filter_kwargs)

            elif len(param_names) > 1:
                self.plot_2d_violin(*attrs, show=show)

        elif len(attrs) == 1:
            self.plot_1d_hist(*attrs, show=show)

        else:
            raise ValueError('Must pass between one and three column names.')


def bind_simulator(**sim_kwargs) -> Callable:
    """
    Create a function with all static simulation data bound to it, where the 
    arguments are simulation parameters
    """

    symbols: List[str] = data_io.get_all_symbols()
    prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

    _hma_trend_signal: Callable = calculate_hma_trend_signal
    _sharpe: Callable = metrics.calculate_rolling_sharpe_ratio

    def _simulate(hma_trend_n: int, sharpe_n: int) -> Performance:
        
        signal = prices.apply(_hma_trend_signal, args=(hma_trend_n,), axis=0)
        signal.iloc[-1] = 0
        preference = prices.apply(_sharpe, args=(sharpe_n, ), axis=0)

        simulator = simulation.SimpleSimulator(**sim_kwargs)
        simulator.simulate(prices, signal, preference)

        return simulator.portfolio_history.get_performance_metric_data()

    return _simulate

if __name__ == '__main__':

    simulate = bind_simulator(initial_cash=10000, max_active_positions=5)

    optimizer = GridSearchOptimizer(simulate)
    optimizer.optimize(
        hma_trend_n=range(10, 110, 10),
        sharpe_n=range(10, 110, 10),
    )

    print(optimizer.get_best('excess_cagr'))
    optimizer.plot('excess_cagr')
    optimizer.plot('hma_trend_n', 'excess_cagr')
    optimizer.plot('hma_trend_n', 'sharpe_n', 'excess_cagr')

# Non parallelized took 161 seconds