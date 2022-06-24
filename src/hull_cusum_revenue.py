import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import os
import pandas as pd

### THIS FILE PLOTS REVENUE EVENTS #####


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MODULE_DIR)
GIT_DIR = os.path.dirname(SRC_DIR)
VQTI_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SRC_DIR, 'data')
EOD_DIR = os.path.join(DATA_DIR, 'eod')
REVENUE_DIR = os.path.join(DATA_DIR, 'revenue')

def load_eod_revenue(symbol: str) -> pd.DataFrame:
	filepath = os.path.join(REVENUE_DIR, f'{symbol}.csv')
	return pd.read_csv(
		filepath, 
		parse_dates=['date'], 
		index_col='date', 
		dtype='float64',
	)
def load_eod(symbol: str) -> pd.DataFrame:
	filepath = os.path.join(EOD_DIR, f'{symbol}.csv')
	return pd.read_csv(
		filepath, 
		parse_dates=['date'], 
		index_col='date', 
		dtype='float64',
	)

def calculate_non_uniform_lagged_change(series: pd.Series, n_days: int):
    """
    Use pd.Series.searchsorted to measure the lagged change in a non-uniformly 
    spaced time series over n_days of calendar time. 
    """

    # Get mapping from now to n_days ago at every point
    _timedelta: pd.Timedelta = pd.Timedelta(days=n_days)
    _idx: pd.Series = series.index.searchsorted(series.index - _timedelta)
    _idx = _idx[_idx > 0]

    # Get the last len(series) - n_days values
    _series = series.iloc[-_idx.shape[0]:]

    # Build a padding of NA values
    _pad_length = series.shape[0] - _idx.shape[0]
    _na_pad = pd.Series(None, index=series.index[:_pad_length])

    # Get the corresonding lagged values
    _lagged_series = series.iloc[_idx]

    # Measure the difference
    _diff = pd.Series(_series.values-_lagged_series.values, index=_series.index)

    return pd.concat([_na_pad, _diff])


def calculate_cusum_events(series: pd.Series, 
    filter_threshold: float) -> pd.DatetimeIndex:
    """
    Calculate symmetric cusum filter and corresponding events
    """

    event_dates = list()
    s_up = 0
    s_down = 0

    for date, price in series.items():
        s_up = max(0, s_up + price)
        s_down = min(0, s_down + price)

        if s_up > filter_threshold:
            s_up = 0
            event_dates.append(date)

        elif s_down < -filter_threshold:
            s_down = 0
            event_dates.append(date)

    return pd.DatetimeIndex(event_dates)

# In pypm.ml_model.events
from pypm import filters

def calculate_events_for_revenue_series(series: pd.Series, 
    filter_threshold: float, lookback: int=365) -> pd.DatetimeIndex:
    """
    Calculate the symmetric cusum filter to generate events on YoY changes in 
    the log revenue series
    """
    series = np.log(series)
    series = filters.calculate_non_uniform_lagged_change(series, lookback)
    return filters.calculate_cusum_events(series, filter_threshold)


def calculate_events(revenue_series: pd.Series):
    return calculate_events_for_revenue_series(
        revenue_series,
        filter_threshold=5,
        lookback=365,
    )

if __name__ == '__main__':

    df = load_eod_revenue('AWU')
    
    stock_price = load_eod('AWU')

    events = calculate_events(df.value)
    
    # Events on Revenue series
    ax = plt.subplot2grid((5,4), (0,0), rowspan=3, colspan=4)
    ax.plot(df.index ,df.value,color='blue',lw=2,label="Close")
    ax.plot(events, df.value[events], '>' , color='red',lw=2, ms=5, label='Events')
    ax.set_title("Events",fontsize=30)
    ax.set_xlabel('Date',fontsize=24)
    ax.set_ylabel('Revenue ($)',fontsize=24)
   
    
    ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=18, facecolor='#D9DDE1')
    ax.grid(color='gray', linestyle='--', linewidth=1)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.show()
    
    
    # Revenue Events On Stock Close Price
    ax = plt.subplot2grid((5,4), (0,0), rowspan=3, colspan=4)
    ax.plot(stock_price.index ,stock_price.close,color='blue',lw=2,label="Close")
    ax.plot(events, stock_price.close[events], '>' , color='red',lw=2, ms=5, label='Events')
    ax.set_title("Events",fontsize=30)
    ax.set_xlabel('Date',fontsize=24)
    ax.set_ylabel('Revenue ($)',fontsize=24)
   
    
    ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=18, facecolor='#D9DDE1')
    ax.grid(color='gray', linestyle='--', linewidth=1)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.show()