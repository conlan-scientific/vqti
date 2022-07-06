import os
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from pypm.data_io import DATA_DIR

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = '\\'.join(os.path.dirname(__file__).split("/"))
VQTI_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(VQTI_DIR, 'data')
EOD_DATA_DIR = os.path.join(DATA_DIR, 'eod')
ALTERNATIVE_DATA_DIR = os.path.join(DATA_DIR, 'revenue')

def load_eod_data(ticker: str, data_dir: str=EOD_DATA_DIR) -> DataFrame:
    f_path = os.path.join(data_dir, f'{ticker}.csv')
    assert os.path.isfile(f_path), f'No data available for {ticker}'
    return pd.read_csv(f_path, parse_dates=['date'], index_col='date')


def load_spy_data() -> DataFrame:
    """
    Convenience function to load S&P 500 ETF EOD data
    """
    return load_eod_data('SPY', DATA_DIR)

def _combine_columns(filepaths_by_symbol: Dict[str, str], 
    attr: str='close') -> pd.DataFrame:

    data_frames = [
        pd.read_csv(
            filepath, 
            index_col='date', 
            usecols=['date', attr], 
            parse_dates=['date'],
        ).rename(
            columns={
                'date': 'date', 
                attr: symbol,
            }
        ) for symbol, filepath in filepaths_by_symbol.items()
    ]
    return pd.concat(data_frames, sort=True, axis=1)    


def load_eod_matrix(tickers: List[str], attr: str='close') -> pd.DataFrame:
    filepaths_by_symbol = {
        t: os.path.join(EOD_DATA_DIR, f'{t}.csv') for t in tickers
    }
    return _combine_columns(filepaths_by_symbol, attr)

def load_alternative_data_matrix(tickers: List[str]) -> pd.DataFrame:
    filepaths_by_symbol = {
        t: os.path.join(ALTERNATIVE_DATA_DIR, f'{t}.csv') for t in tickers
    }
    return _combine_columns(filepaths_by_symbol, 'value')

def load_volume_matrix(tickers: List[str]) -> pd.DataFrame:
    filepaths_by_symbol = {
        t: os.path.join(EOD_DATA_DIR, f'{t}.csv') for t in tickers
    }
    return _combine_columns(filepaths_by_symbol, 'volume')

def get_all_symbols() -> List[str]:
    return [v.strip('.csv') for v in os.listdir(EOD_DATA_DIR)]


def build_eod_closes() -> None:
    filenames = os.listdir(EOD_DATA_DIR)
    filepaths_by_symbol = {
        v.strip('.csv'): os.path.join(EOD_DATA_DIR, v) for v in filenames
    }
    result = _combine_columns(filepaths_by_symbol)
    result.to_csv(os.path.join(DATA_DIR, 'eod_closes.csv'))


def concatenate_metrics(df_by_metric: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates different dataframes that have the same columns into a
    hierarchical dataframe.

    The input df_by_metric should of the form

    {
        'metric_1': pd.DataFrame()
        'metric_2: pd.DataFrame()
    }
    where each dataframe should have the same columns, i.e. symbols.
    """

    to_concatenate = []
    tuples = []
    for key, df in df_by_metric.items():
        to_concatenate.append(df)
        tuples += [(s, key) for s in df.columns.values]

    df = pd.concat(to_concatenate, sort=True, axis=1)
    df.columns = pd.MultiIndex.from_tuples(tuples, names=['symbol', 'metric'])

    return df

def load_data() -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
	"""
	Load the data as is will be used in the alternative data model
	"""
	symbols: List[str] = get_all_symbols()
	alt_data = load_alternative_data_matrix(symbols)
	eod_data = load_eod_matrix(symbols)
	eod_data = eod_data[eod_data.index >= alt_data.index.min()]

	return symbols, eod_data, alt_data

def load_volume_data() -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    Load the data as is will be used in the alternative data model
    """
    symbols: List[str] = get_all_symbols()
    volume_data = load_volume_matrix(symbols)
    eod_data = load_eod_matrix(symbols)
    eod_data = eod_data[eod_data.index >= volume_data.index.min()]
    
    return symbols, eod_data, volume_data

def load_volume_and_revenue_data() -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    Load the data as is will be used in the alternative data model
    """
    symbols: List[str] = get_all_symbols()
    volume_data = load_volume_matrix(symbols)
    revenue_data = load_alternative_data_matrix(symbols)
    eod_data = load_eod_matrix(symbols)
    volume_data = volume_data[volume_data.index >= revenue_data.index.min()]
    eod_data = eod_data[eod_data.index >= revenue_data.index.min()]
    
    return symbols, eod_data, volume_data, revenue_data

from pypm import labels

def calculate_tbm_labels(price_series, event_index) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate labels based on the triple barrier method. Return a series of 
    event labels index by event start date, and return a series of event end 
    dates indexed by event start date.
    """

    # Remove event that don't have a proper chance to materialize
    time_delta_days = 90
    max_date = price_series.index.max()
    cutoff = max_date - pd.Timedelta(days=time_delta_days)
    event_index = event_index[event_index <= cutoff]

    # Use triple barrier method
    event_labels, event_spans = labels.compute_triple_barrier_labels(
        price_series,
        event_index,
        time_delta_days=time_delta_days,
        #upper_delta=0.10,
        #lower_delta=-0.10,
        upper_z=1.8,
        lower_z=-1.8,
        lower_label=-1,
    )

    return event_labels, event_spans

from pypm import indicators, filters, metrics
from vqti.indicators.hma import calculate_numpy_matrix_hma
from vqti.indicators.hma_signals import calculate_hma_zscore

_calc_delta = filters.calculate_non_uniform_lagged_change
_calc_ma = indicators.calculate_simple_moving_average
_calc_log_return = metrics.calculate_log_return_series

def _calc_rolling_vol(series, n):
    return series.rolling(n).std() * np.sqrt(252 / n)

def calculate_hull_features(price_series, volume_series, revenue_series) -> pd.DataFrame:
    """
    Calculate any and all potentially useful features. Return as a dataframe.
    """
    log_volume = np.log(volume_series)
    log_revenue = np.log(revenue_series)
    log_prices = np.log(price_series)

    log_revenue_ma = _calc_ma(log_revenue, 10)
    log_prices_ma = _calc_ma(log_prices, 10)
    log_volume_ma = _calc_ma(log_volume, 10)
    
    log_returns = _calc_log_return(price_series)

    features_by_name = dict()

    for i in [7, 30, 90, 180, 360]:

        rev_feature = _calc_delta(log_revenue_ma, i)
        price_feature = _calc_delta(log_prices_ma, i)
        vol_feature = _calc_rolling_vol(log_returns, i)
        volume_feature = _calc_delta(log_volume_ma, i)
        #volume_feature = _calc_rolling_vol(log_volume_ma, i)
        features_by_name.update({
            f'{i}_day_revenue_delta': rev_feature,
            f'{i}_day_return': price_feature,
            f'{i}_day_vol': vol_feature,
            f'{i}_day_volume_delta' : volume_feature,
        })
        
    hma_trend_10 = calculate_numpy_matrix_hma(log_prices, 16)
    hma_trend_25 = calculate_numpy_matrix_hma(log_prices, 25)
    hma_trend_49 = calculate_numpy_matrix_hma(log_prices, 49)
    hma_trend_81 = calculate_numpy_matrix_hma(log_prices, 81)

    hma_zscore_25_49 = calculate_hma_zscore(log_prices,25,49)
    
    features_by_name.update({
        'hma_trend_10': hma_trend_10,
		'hma_trend_25': hma_trend_25,
		'hma_trend_49': hma_trend_49,
		'hma_trend_81': hma_trend_81,
        'hma_zscore_25_49': hma_zscore_25_49,
    })
        
    features_df = pd.DataFrame(features_by_name)    
    return features_df



def plot_features_hist(features_df: pd.DataFrame):
    for i in range(len(features_df.columns[:-1])):
        label = features_df.columns[i]
        plt.hist(features_df[features_df['y']==1][label], color='blue', 
                 label="Winning Trade", alpha=0.7, density=True, bins=15)
        plt.hist(features_df[features_df['y']==0][label], color='red', 
                 label=" Losing Trade", alpha=0.7, density=True, bins=15)
        plt.title(label)
        plt.ylabel("Probability")
        plt.xlabel(label)
        plt.legend()
        plt.show()
    return
def plot_scaled_features_df(df: pd.DataFrame):
    
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X.shape, y.shape
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    X.shape, y.shape
    transformed_df = pd.DataFrame(data, columns=df.columns)
    
    for i in range(len(transformed_df.columns[:-1])):
        label = transformed_df.columns[i]
        plt.hist(transformed_df[transformed_df['y']==1][label], color='blue', 
                 label="Winning Trade", alpha=0.7, density=True, bins=15)
        plt.hist(transformed_df[transformed_df['y']==0][label], color='red', 
                 label=" Losing Trade", alpha=0.7, density=True, bins=15)
        plt.title(label)
        plt.ylabel("Probability")
        plt.xlabel(label)
        plt.legend()
        plt.show()
    return


if __name__ == '__main__':
    build_eod_closes()