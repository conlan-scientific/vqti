from RSI_jake import pure_python_relative_stength_index, signal_line_calculation
from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List 

#get all data somehow, not sure :(

def _calculate_return_series(self, series: pd.Series) -> pd.Series:
    """
    From Algorithmic Trading with Python
    """
    shifted_series = series.shift(1, axis=0)
    return series / shifted_series - 1

def _get_years_passed(self, series: pd.Series) -> float:
    """
    From Algorithmic Trading with Python p.24
    """

    start_date = series.index[0]
    end_date = series.index[-1]
    return (end_date - start_date).days / 365.25

def _calculate_cagr(self, return_series: pd.Series) -> float:
    """
    From Algorithmic Trading with Python
    """
    value_factor = return_series.iloc[-1] / return_series.iloc[0]
    years_passed = self._get_years_passed(return_series)
    return (value_factor ** (1 / years_passed)) - 1

def _calculate_annualized_volatility(self, series: pd.Series) -> float:
    """
    From Algorithmic Trading with Python p.24
    """
    years_passed = self._get_years_passed(series)
    entries_per_year = series.shape[0] / years_passed
    return series.std() * np.sqrt(entries_per_year)

def _calculate_sharpe_ratio(self, price_series: pd.Series, benchmark_rate: float = 0) -> float:
    """
    From Algorithmic Trading with Python p.27
    """
    cagr = self._calculate_cagr(price_series)
    return_series = self._calculate_return_series(price_series)
    volatility = self._calculate_annualized_volatility(return_series)
    return (cagr - benchmark_rate) / volatility

def full_simulation():
    prices_df = get_all_data()

    for (col_name, colval) in prices_df.iteritems():
        currentColRSI = pure_python_relative_stength_index(colval, len(colval))
        prices_df[col_name] = currentColRSI

    for (col_name, colval) in prices_df.iteritems():
        currentColSignalLine = signal_line_calculation(colval)
        prices_df[col_name] = currentColSignalLine

    max_assets = 20
    df_index = prices_df.index
    cash = starting_cash = 100000
    portfolio: List[str] = list()
    equity_curve = dict()
    original_price_df = get_all_data()


    for date in df_index:
        signals = prices_df.loc[date]

        stocks_im_going_to_buy: List[str] = signals[signals == 1].index.tolist()
        stocks_im_going_to_sell: List[str] = signals[signals == -1].index.tolist()
        stocks_im_holding: List[str] = portfolio

        for stock in stocks_im_going_to_sell:
            #how do we determine how mnay of each stock is in the portfolio, wouldn't a dictionary work better than a list?
            if stock in portfolio:
                portfolio.pop(stock)
                stocks_im_holding.pop(stock)
                cash += original_price_df[date][stock]
            
        for stock in stocks_im_going_to_buy:
            cash_to_spend = cash / (max_assets - len(stocks_im_holding))
            #I'm not sure how we're dividing up money between stocks, even for all 20 stocks?
            if stock not in portfolio:
                if len(stocks_im_holding) < max_assets:
                    portfolio.append(stock)
                    stocks_im_holding.append(stock)
            # else:
                    #how to deal with overflow??? Which stocks do I want to keep? Alphabetical maybe

        current_portfolio_value = 0
        for stock in portfolio:
            current_portfolio_value += original_price_df[date][stock]

        equity_curve[date] = cash + current_portfolio_value
    equity_curve.plot()
    plt.show()
    
    sharpe_ratio = _calculate_sharpe_ratio(pd.to_datetime(equity_curve.index))