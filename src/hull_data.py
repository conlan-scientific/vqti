import os
import pandas as pd
from pandas import DataFrame
from typing import Dict, List, Tuple

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


if __name__ == '__main__':
    build_eod_closes()