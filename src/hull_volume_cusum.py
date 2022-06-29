
import numpy as np
import pandas as pd
from vqti.load import load_eod
import matplotlib.pyplot as plt


#### THIS FILE CALCULATES EVENTS BASED ON VOLUME CHANGE AND ACCUMULATION FROM THE DAY BEFORE #######

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
    _na_pad = pd.Series(None, index=series.index[:_pad_length], dtype=np.float64)

    # Get the corresonding lagged values
    _lagged_series = series.iloc[_idx]

    # Measure the difference
    _diff= pd.Series((_series.values-_lagged_series.values)/ \
        np.abs(_lagged_series.values), index=_series.index)
    return pd.concat([_na_pad, _diff])


def calculate_cusum_events(series: pd.Series, 
    filter_threshold: float) -> pd.DatetimeIndex:
    """
    Calculate symmetric cusum filter and corresponding events
    """

    event_dates = list()
    s_up = 0
    s_down = 0

    for date, volume in series.items():
        s_up = max(0, s_up + volume)
        s_down = min(0, s_down + volume)

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
    filter_threshold: float, lookback: int=1) -> pd.DatetimeIndex:
    """
    Calculate the symmetric cusum filter to generate events on day-to-day changes in 
    the log volume series
    """
    #series = pd.Series.pct_change(series)
    #series = np.log(series)
    series = calculate_non_uniform_lagged_change(series, lookback)
    return calculate_cusum_events(series, filter_threshold)


def calculate_volume_events(volume_series: pd.Series):
    return calculate_events_for_revenue_series(
        volume_series,
        filter_threshold=2.0,
        lookback=1,
    )


if __name__ == '__main__':

    df = load_eod('AWU')

    events = calculate_volume_events(df.volume)
   
    ax = plt.subplot2grid((5,4), (0,0), rowspan=3, colspan=4)
    ax.plot(df.index ,df.close,color='blue',lw=2,label="Close")
    ax.plot(events, df.close[events], '>' , color='red',lw=2, ms=5, 
            label='Events')
    ax.set_title("Events",fontsize=30)
    ax.set_xlabel('Date',fontsize=24)
    ax.set_ylabel('Price ($)',fontsize=24)
    bottom_plt = plt.subplot2grid((5,4), (3,0), rowspan=1, colspan=4)
    bottom_plt.bar(df.index, df.volume)
    
    ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, 
              fontsize=18, facecolor='#D9DDE1')
    ax.grid(color='gray', linestyle='--', linewidth=1)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.show()
    
   



        