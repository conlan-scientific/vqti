from re import X
from signal import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Callable, NewType, Any, Iterable, Set
from collections import OrderedDict, defaultdict
from vqti.load import EOD_DIR, load_eod
import os
import glob
from hullma import numpy_matrix_hma
from hullma_signal import hma_trend_signal

def hma_trend_signal(series: pd.Series, m: int=16) -> pd.Series:
    hull_ma = pd.Series(numpy_matrix_hma(series.values, m))
    trend = np.sign(hull_ma - hull_ma.shift(1))
    signal = np.where(trend > trend.shift(1), 1, 0)
    signal = np.where(trend < trend.shift(1), -1, signal)
    return signal


from pypm import metrics, signals, data_io, portfolio

Symbol = NewType('Symbol', str)
Dollars = NewType('Dollars', float)

DATE_FORMAT_STR = '%a %b %d, %Y'
def _pdate(date: pd.Timestamp):
    """Pretty-print a datetime with just the date"""
    return date.strftime(DATE_FORMAT_STR)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MODULE_DIR)
GIT_DIR = os.path.dirname(SRC_DIR)
VQTI_DIR = os.path.join(GIT_DIR, 'VQTI')
DATA_DIR = os.path.join(VQTI_DIR, 'data')
EOD_DIR = os.path.join(DATA_DIR, 'eod')
REVENUE_DIR = os.path.join(DATA_DIR, 'revenue')

def load_eod(symbol: str) -> pd.DataFrame:
	filepath = os.path.join(EOD_DIR, f'{symbol}.csv')
	return pd.read_csv(
		filepath, 
		parse_dates=['date'], 
		index_col='date', 
		dtype='float64',
	)



class Position(object):
    """
    A simple object to hold and manipulate data related to long stock trades.
    Allows a single buy and sell operation on an asset for a constant number of 
    shares.
    The __init__ method is equivalent to a buy operation. The exit
    method is a sell operation.
    """

    def __init__(self, symbol: Symbol, entry_date: pd.Timestamp, 
        entry_price: Dollars, shares: int):
        """
        Equivalent to buying a certain number of shares of the asset
        """

        # Recorded on initialization
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.symbol = symbol

        # Recorded on position exit
        self.exit_date: pd.Timestamp = None
        self.exit_price: Dollars = None

        # For easily getting current portfolio value
        self.last_date: pd.Timestamp = None
        self.last_price: Dollars = None

        # Updated intermediately
        self._dict_series: Dict[pd.Timestamp, Dollars] = OrderedDict()
        self.record_price_update(entry_date, entry_price)

        # Cache control for pd.Series representation
        self._price_series: pd.Series = None
        self._needs_update_pd_series: bool = True

    def exit(self, exit_date, exit_price):
        """
        Equivalent to selling a stock holding
        """
        assert self.entry_date != exit_date, 'Churned a position same-day.'
        assert not self.exit_date, 'Position already closed.'
        self.record_price_update(exit_date, exit_price)
        self.exit_date = exit_date
        self.exit_price = exit_price

    def record_price_update(self, date, price):
        """
        Stateless function to record intermediate prices of existing positions
        """
        self.last_date = date
        self.last_price = price
        self._dict_series[date] = price

        # Invalidate cache on self.price_series
        self._needs_update_pd_series = True

    @property
    def price_series(self) -> pd.Series:
        """
        Returns cached readonly pd.Series 
        """
        if self._needs_update_pd_series or self._price_series is None:
            self._price_series = pd.Series(self._dict_series)
            self._needs_update_pd_series = False
        return self._price_series

    @property
    def last_value(self) -> Dollars:
        return self.last_price * self.shares

    @property
    def is_active(self) -> bool:
        return self.exit_date is None

    @property
    def is_closed(self) -> bool:
        return not self.is_active
    
    @property
    def value_series(self) -> pd.Series:
        """
        Returns the value of the position over time. Ignores self.exit_date.
        Used in calculating the equity curve.
        """
        assert self.is_closed, 'Position must be closed to access this property'
        return self.shares * self.price_series[:-1]

    @property
    def percent_return(self) -> float:
        return (self.exit_price / self.entry_price) - 1
    
    @property
    def entry_value(self) -> Dollars:
        return self.shares * self.entry_price

    @property
    def exit_value(self) -> Dollars:
        return self.shares * self.exit_price

    @property
    def change_in_value(self) -> Dollars:
        return self.exit_value - self.entry_value

    @property
    def trade_length(self):
        return len(self._dict_series) - 1
    
    def print_position_summary(self):
        _entry_date = _pdate(self.entry_date)
        _exit_date = _pdate(self.exit_date)
        _days = self.trade_length

        _entry_price = round(self.entry_price, 2)
        _exit_price = round(self.exit_price, 2)

        _entry_value = round(self.entry_value, 2)
        _exit_value = round(self.exit_value, 2)

        _return = round(100 * self.percent_return, 1)
        _diff = round(self.change_in_value, 2)

        print(f'{self.symbol:<5}     Trade summary')
        print(f'Date:     {_entry_date} -> {_exit_date} [{_days} days]')
        print(f'Price:    ${_entry_price} -> ${_exit_price} [{_return}%]')
        print(f'Value:    ${_entry_value} -> ${_exit_value} [${_diff}]')
        print()

    def __hash__(self):
        """
        A unique position will be defined by a unique combination of an 
        entry_date and symbol, in accordance with our constraints regarding 
        duplicate, variable, and compound positions
        """
        return hash((self.entry_date, self.symbol))

class PortfolioHistory(object):
    """
    Holds Position objects and keeps track of portfolio variables.
    Produces summary statistics.
    """

    def __init__(self):
        # Keep track of positions, recorded in this list after close
        self.position_history: List[Position] = []
        self._logged_positions: Set[Position] = set()

        # Keep track of the last seen date
        self.last_date: pd.Timestamp = pd.Timestamp.min

        # Readonly fields
        self._cash_history: Dict[pd.Timestamp, Dollars] = dict()
        self._simulation_finished = False
        self._spy: pd.DataFrame = pd.DataFrame()
        self._spy_log_returns: pd.Series = pd.Series()

    def add_to_history(self, position: Position):
        _log = self._logged_positions
        assert not position in _log, 'Recorded the same position twice.'
        assert position.is_closed, 'Position is not closed.'
        self._logged_positions.add(position)
        self.position_history.append(position)
        self.last_date = max(self.last_date, position.last_date)

    def record_cash(self, date, cash):
        self._cash_history[date] = cash
        self.last_date = max(self.last_date, date)

    @staticmethod
    def _as_oseries(d: Dict[pd.Timestamp, Any]) -> pd.Series:
        return pd.Series(d).sort_index()

    def _compute_cash_series(self):
        self._cash_series = self._as_oseries(self._cash_history)

    @property
    def cash_series(self) -> pd.Series:
        return self._cash_series

    def _compute_portfolio_value_series(self):
        value_by_date = defaultdict(float)
        last_date = self.last_date

        # Add up value of assets
        for position in self.position_history:
            for date, value in position.value_series.items():
                value_by_date[date] += value

        # Make sure all dates in cash_series are present
        for date in self.cash_series.index:
            value_by_date[date] += 0

        self._portfolio_value_series = self._as_oseries(value_by_date)

    @property
    def portfolio_value_series(self):
        return self._portfolio_value_series

    def _compute_equity_series(self):
        c_series = self.cash_series
        p_series = self.portfolio_value_series
        assert all(c_series.index == p_series.index), \
            'portfolio_series has dates not in cash_series'
        self._equity_series = c_series + p_series     

    @property
    def equity_series(self):
        return self._equity_series

    def _compute_log_return_series(self):
        self._log_return_series = \
            metrics.calculate_log_return_series(self.equity_series)

    @property
    def log_return_series(self):
        return self._log_return_series

    def _assert_finished(self):
        assert self._simulation_finished, \
            'Simuation must be finished by running self.finish() in order ' + \
            'to access this method or property.'

    def finish(self):
        """
        Notate that the simulation is finished and compute readonly values
        """
        self._simulation_finished = True
        self._compute_cash_series()
        self._compute_portfolio_value_series()
        self._compute_equity_series()
        self._compute_log_return_series()
        self._assert_finished()

    def compute_portfolio_size_series(self) -> pd.Series:
        size_by_date = defaultdict(int)
        for position in self.position_history:
            for date in position.value_series.index:
                size_by_date[date] += 1
        return self._as_oseries(size_by_date)

    @property
    def spy(self):
        if self._spy.empty:
            self._spy = data_io.load_spy_data()
        return self._spy

    @property
    def spy_log_returns(self):
        if self._spy_log_returns.empty:
            close = self.spy['close']
            self._spy_log_returns =  metrics.calculate_log_return_series(close)
        return self._spy_log_returns

    @property
    def percent_return(self):
        return metrics.calculate_percent_return(self.equity_series)

    @property
    def spy_percent_return(self):
        return metrics.calculate_percent_return(self.spy['close'])

    @property
    def cagr(self):
        return metrics.calculate_cagr(self.equity_series)

    @property
    def volatility(self):
        return metrics.calculate_annualized_volatility(self.log_return_series)

    @property
    def sharpe_ratio(self):
        return metrics.calculate_sharpe_ratio(self.equity_series)

    @property
    def spy_cagr(self):
        return metrics.calculate_cagr(self.spy['close'])
    
    @property
    def excess_cagr(self):
        return self.cagr - self.spy_cagr

    @property
    def jensens_alpha(self):
        return metrics.calculate_jensens_alpha(
            self.log_return_series,
            self.spy_log_returns,
        )

    @property
    def dollar_max_drawdown(self):
        return metrics.calculate_max_drawdown(self.equity_series, 'dollar')

    @property
    def percent_max_drawdown(self):
        return metrics.calculate_max_drawdown(self.equity_series, 'percent')

    @property
    def log_max_drawdown_ratio(self):
        return metrics.calculate_log_max_drawdown_ratio(self.equity_series)
    
    @property
    def number_of_trades(self):
        return len(self.position_history)

    @property
    def average_active_trades(self):
        return self.compute_portfolio_size_series().mean()

    @property
    def final_cash(self):
        self._assert_finished()
        return self.cash_series[-1]
    
    @property
    def final_equity(self):
        self._assert_finished()
        return self.equity_series[-1]
    
    def print_position_summaries(self):
        for position in self.position_history:
            position.print_position_summary()

    def print_summary(self):
        self._assert_finished()
        s = f'Equity: ${self.final_equity:.2f}\n' \
            f'Percent Return: {100*self.percent_return:.2f}%\n' \
            f'S&P 500 Return: {100*self.spy_percent_return:.2f}%\n\n' \
            f'Number of trades: {self.number_of_trades}\n' \
            f'Average active trades: {self.average_active_trades:.2f}\n\n' \
            f'CAGR: {100*self.cagr:.2f}%\n' \
            f'S&P 500 CAGR: {100*self.spy_cagr:.2f}%\n' \
            f'Excess CAGR: {100*self.excess_cagr:.2f}%\n\n' \
            f'Annualized Volatility: {100*self.volatility:.2f}%\n' \
            f'Sharpe Ratio: {self.sharpe_ratio:.2f}\n' \
            f'Jensen\'s Alpha: {self.jensens_alpha:.6f}\n\n' \
            f'Dollar Max Drawdown: ${self.dollar_max_drawdown:.2f}\n' \
            f'Percent Max Drawdown: {100*self.percent_max_drawdown:.2f}%\n' \
            f'Log Max Drawdown Ratio: {self.log_max_drawdown_ratio:.2f}\n'

        print(s)

    def plot(self, show=True) -> plt.Figure:
        """
        Plots equity, cash and portfolio value curves.
        """
        self._assert_finished()

        figure, axes = plt.subplots(nrows=3, ncols=1)
        figure.tight_layout(pad=3.0)
        axes[0].plot(self.equity_series)
        axes[0].set_title('Equity')
        axes[0].grid()

        axes[1].plot(self.cash_series)
        axes[1].set_title('Cash')
        axes[1].grid()

        axes[2].plot(self.portfolio_value_series)
        axes[2].set_title('Portfolio Value')
        axes[2].grid()

        if show:
            plt.show()

        return figure

class SimpleSimulator(object):
    """
    A simple trading simulator to work with the PortfolioHistory class
    """

    def __init__(self, initial_cash: float=10000, max_active_positions: int=5,
        percent_slippage: float=0.0005, trade_fee: float=1):

        ### Set simulation parameters

        # Initial cash in portfolio
        # self.cash will fluctuate
        self.initial_cash = self.cash = initial_cash

        # Maximum number of different assets that can be help simultaneously
        self.max_active_positions: int = max_active_positions

        # The percentage difference between closing price and fill price for the
        # position, to simulate adverse effects of market orders
        self.percent_slippage = percent_slippage

        # The fixed fee in order to open a position in dollar terms
        self.trade_fee = trade_fee

        # Keep track of live trades
        self.active_positions_by_symbol: Dict[Symbol, Position] = OrderedDict()

        # Keep track of portfolio history like cash, equity, and positions
        self.portfolio_history = PortfolioHistory()

    @property
    def active_positions_count(self):
        return len(self.active_positions_by_symbol)

    @property
    def free_position_slots(self):
        return self.max_active_positions - self.active_positions_count

    @property
    def active_symbols(self) -> List[Symbol]:
        return list(self.active_positions_by_symbol.keys())

    def print_initial_parameters(self):
        s = f'Initial Cash: ${self.initial_cash} \n' \
            f'Maximum Number of Assets: {self.max_active_positions}\n'
        print(s)
        return s

    @staticmethod
    def make_tuple_lookup(columns) -> Callable[[str, str], int]:
        """
        Map a multi-index dataframe to an itertuples-like object.
        The index of the dateframe is always the zero-th element.
        """

        # col is a hierarchical column index represented by a tuple of strings
        tuple_lookup: Dict[Tuple[str, str], int] = { 
            col: i + 1 for i, col in enumerate(columns) 
        }

        return lambda symbol, metric: tuple_lookup[(symbol, metric)]

    @staticmethod
    def make_all_valid_lookup(_idx: Callable):
        """
        Return a function that checks for valid data, given a lookup function
        """
        return lambda row, symbol: (
            not pd.isna(row[_idx(symbol, 'pref')]) and \
            not pd.isna(row[_idx(symbol, 'signal')]) and \
            not pd.isna(row[_idx(symbol, 'price')])
        )

    def buy_to_open(self, symbol, date, price):
        """
        Keep track of new position, make sure it isn't an existing position. 
        Verify you have cash.
        """

        # Figure out how much we are willing to spend
        cash_to_spend = self.cash / self.free_position_slots
        cash_to_spend -= self.trade_fee

        # Calculate buy_price and number of shares. Fractional shares allowed.
        purchase_price = (1 + self.percent_slippage) * price
        shares = cash_to_spend / purchase_price

        # Spend the cash
        self.cash -= cash_to_spend + self.trade_fee
        assert self.cash >= 0, 'Spent cash you do not have.'
        self.portfolio_history.record_cash(date, self.cash)   

        # Record the position
        positions_by_symbol = self.active_positions_by_symbol
        assert not symbol in positions_by_symbol, 'Symbol already in portfolio.'        
        position = Position(symbol, date, purchase_price, shares)
        positions_by_symbol[symbol] = position

    def sell_to_close(self, symbol, date, price):
        """
        Keep track of exit price, recover cash, close position, and record it in
        portfolio history.
        Will raise a KeyError if symbol isn't an active position
        """

        # Exit the position
        positions_by_symbol = self.active_positions_by_symbol
        position = positions_by_symbol[symbol]
        position.exit(date, price)

        # Receive the cash
        sale_value = position.last_value * (1 - self.percent_slippage)
        self.cash += sale_value
        self.portfolio_history.record_cash(date, self.cash)

        # Record in portfolio history
        self.portfolio_history.add_to_history(position)
        del positions_by_symbol[symbol]
    
    @staticmethod
    def _assert_equal_columns(*args: Iterable[pd.DataFrame]):
        column_names = set(args[0].columns.values)
        for arg in args[1:]:
            assert set(arg.columns.values) == column_names, \
                'Found unequal column names in input data frames.'

    def simulate(self, price: pd.DataFrame, signal: pd.DataFrame, 
        preference: pd.DataFrame):
        """
        Runs the simulation.
        price, signal, and preference are data frames with the column names 
        represented by the same set of stock symbols.
        """

        # Create a hierarchical data frame to loop through
        self._assert_equal_columns(price, signal, preference)
        df = data_io.concatenate_metrics({
            'price': price,
            'signal': signal,
            'pref': preference,
        })

        # Get list of symbols
        all_symbols = list(set(price.columns.values))

        # Get lookup functions
        _idx = self.make_tuple_lookup(df.columns)
        _all_valid = self.make_all_valid_lookup(_idx)

        # Store some variables
        active_positions_by_symbol = self.active_positions_by_symbol
        max_active_positions = self.max_active_positions

        # Iterating over all dates.
        # itertuples() is significantly faster than iterrows(), it however comes
        # at the cost of being able index easily. In order to get around this
        # we use an tuple lookup function: "_idx"
        for row in df.itertuples():

            # date index is always first element of tuple row
            date = row[0]

            # Get symbols with valid and tradable data
            symbols: List[str] = [s for s in all_symbols if _all_valid(row, s)]

            # Iterate over active positions and sell stocks with a sell signal.
            _active = self.active_symbols
            to_exit = [s for s in _active if row[_idx(s, 'signal')] == -1]
            for s in to_exit:
                sell_price = row[_idx(s, 'price')]
                self.sell_to_close(s, date, sell_price)

            # Get up to max_active_positions symbols with a buy signal in 
            # decreasing order of preference
            to_buy = [
                s for s in symbols if \
                    row[_idx(s, 'signal')] == 1 and \
                    not s in active_positions_by_symbol
            ]
            to_buy.sort(key=lambda s: row[_idx(s, 'pref')], reverse=True)
            to_buy = to_buy[:max_active_positions]

            for s in to_buy:
                buy_price = row[_idx(s, 'price')]
                buy_preference = row[_idx(s, 'pref')]

                # If we have some empty slots, just buy the asset outright
                if self.active_positions_count < max_active_positions:
                    self.buy_to_open(s, date, buy_price)
                    continue

                # If are holding max_active_positions, evaluate a swap based on
                # preference
                _active = self.active_symbols
                active_prefs = [(s, row[_idx(s, 'pref')]) for s in _active]

                _min = min(active_prefs, key=lambda k: k[1])
                min_active_symbol, min_active_preference = _min

                # If a more preferable symbol exists, then sell an old one
                if min_active_preference < buy_preference:
                    sell_price = row[_idx(min_active_symbol, 'price')]
                    self.sell_to_close(min_active_symbol, date, sell_price)
                    self.buy_to_open(s, date, buy_price)

            # Update price data everywhere
            for s in self.active_symbols:
                price = row[_idx(s, 'price')]
                position = active_positions_by_symbol[s]
                position.record_price_update(date, price)

        # Sell all positions and mark simulation as complete
        for s in self.active_symbols:
            self.sell_to_close(s, date, row[_idx(s, 'price')])
        self.portfolio_history.finish()
    
if __name__ == '__main__':

    symbol = 'AWU'
    df = load_eod(symbol)
    shares_to_buy = 50
    df['signal'] = hma_trend_signal(df.close, 16)
    print(df)
    for i, row in enumerate(df.itertuples()):
        date = row.Index
        price = row.close
        s = row.signal
        stocks_im_holding = []

        if s == 1:
            pos = Position(symbol, date, price, shares_to_buy)
            stocks_im_holding.append(pos)
        if not stocks_im_holding:
            pass
        else:
            if s == 0:
                pos.record_price_update(date, price)
            elif s == -1:
                pos.exit(date, price)

    pos.print_position_summary()

    