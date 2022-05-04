import os
import pandas as pd
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MODULE_DIR)
GIT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(GIT_DIR, 'data')
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


if __name__ == '__main__':
	df = load_eod('AWU')
