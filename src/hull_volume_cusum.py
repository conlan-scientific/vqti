
import numpy as np
import pandas as pd
from vqti.load import load_eod
import matplotlib.pyplot as plt


#### THIS FILE CALCULATES EVENTS BASED ON VOLUME CHANGE FROM THE DAY BEFORE #######

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
    #_diff = pd.Series(_series.values-_lagged_series.values, index=_series.index)
    _diff2= pd.Series((_series.values-_lagged_series.values)/ np.abs(_lagged_series.values), index=_series.index)
    return pd.concat([_na_pad, _diff2])


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
   
    _plot(series, filter_threshold, event_dates, s_up, s_down)
        
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
    series = np.log(series)
    series = filters.calculate_non_uniform_lagged_change(series, lookback)
    return filters.calculate_cusum_events(series, filter_threshold)


def calculate_volume_events(volume_series: pd.Series):
    return calculate_events_for_revenue_series(
        volume_series,
        filter_threshold=2.0,
        lookback=1,
    )

def _plot(series, threshold, event_dates, s_up, s_down):
    """Plot results of the detect_cusum function, see its help."""

    (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    t = range(series.size)
    ax1.plot(t, series, 'b-', lw=2)
    if len(event_dates):
        ax1.plot(event_dates, series[event_dates], '>', mfc='g', mec='g', ms=10,
                    label='Event Start')
            
    ax1.set_xlim(-.01*series.size, series.size*1.01-1)
    ax1.set_xlabel('Data #', fontsize=14)
    ax1.set_ylabel('Amplitude', fontsize=14)
    ymin, ymax = series[np.isfinite(x)].min(), series[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax1.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
    ax1.set_title('Time series and Detected Events ' +
                      '(threshold= %.3g): N changes = %d'
                      % (threshold, len(event_dates)))
    ax2.plot(t, s_up, 'y-', label='+')
    ax2.plot(t, s_down, 'm-', label='-')
    ax2.set_xlim(-.01*series.size, series.size*1.01-1)
    ax2.set_xlabel('Data #', fontsize=14)
    ax2.set_ylim(-0.01*threshold, 1.1*threshold)
    ax2.axhline(threshold, color='r')
    ax1.set_ylabel('Amplitude', fontsize=14)
    ax2.set_title('Time series of the cumulative sums of ' +
                      'positive and negative changes')
    ax2.legend(loc='best', framealpha=.5, numpoints=1)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    df = load_eod('AWU')

    events = calculate_volume_events(df.volume)
   
    ax = plt.subplot2grid((5,4), (0,0), rowspan=3, colspan=4)
    ax.plot(df.index ,df.close,color='blue',lw=2,label="Close")
    ax.plot(events, df.close[events], '>' , color='red',lw=2, ms=5, label='Events')
    ax.set_title("Events",fontsize=30)
    ax.set_xlabel('Date',fontsize=24)
    ax.set_ylabel('Price ($)',fontsize=24)
    bottom_plt = plt.subplot2grid((5,4), (3,0), rowspan=1, colspan=4)
    bottom_plt.bar(df.index, df.volume)
    
    ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=18, facecolor='#D9DDE1')
    ax.grid(color='gray', linestyle='--', linewidth=1)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.show()
    
   



        