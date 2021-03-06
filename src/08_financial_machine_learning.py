import pandas as pd
from vqti.load import load_eod
from pypm.data_io import get_all_symbols
from vqti.indicators.cci import pandas_cci_rolling
from vqti.indicators.aroon_oscillator import aroon_pandas

from sklearn import tree
from sklearn import ensemble
from vqti.indicators.bollinger import calculate_percent_b
import numpy as np

symbols = get_all_symbols() # ['AWU', 'BGH', ...]

_dfs = list()
for symbol in symbols:
	print('Preparing training data for', symbol, '...')
	df = load_eod(symbol)

	# Trade the stock once every 22 trading days (monthly)
	t0: pd.Index = df.index[::22]

	# Exit after 365 calendar days
	t1 = pd.Series(
		df.index[[min(df.shape[0]-1, x) for x in df.index.searchsorted(t0 + pd.Timedelta(days=365))]], 
		index=t0,
	)

	# A +1 if the stock goes up and a 0 if it goes down
	# This has length equal to t0 and t1
	# TODO: Do a better job at setting y to something really significant
	# TODO: Investigate the triple barrier method
	assert t0.shape == t1.shape
	# _y = (df.close[t1].values > df.close[t0].values).astype('int64')
	# _y = pd.Series(_y, index=t0, name='y')

	_y = [
		(1 if x > 0.20 else (-1 if x < 0 else 0)) for x in \
		(df.close[t1].values / df.close[t0].values - 1)
	]
	_y = pd.Series(_y, index=t0, name='y')
	t0 = t0[_y != 0]
	t1 = t1[_y != 0]


	_X = pd.DataFrame({
		# 'cci__10': pandas_cci_rolling(df.close, window=10),
		# 'cci__20': pandas_cci_rolling(df.close, window=20),
		# 'cci__40': pandas_cci_rolling(df.close, window=40),
		# 'cci__80': pandas_cci_rolling(df.close, window=80),

		# 'aroon__10': aroon_pandas(df.high, df.low, p=10),
		# 'aroon__20': aroon_pandas(df.high, df.low, p=20),
		# 'aroon__40': aroon_pandas(df.high, df.low, p=40),
		# 'aroon__80': aroon_pandas(df.high, df.low, p=80),

		'bollinger_percent_b__10': calculate_percent_b(df.close, n=10),
		'bollinger_percent_b__20': calculate_percent_b(df.close, n=20),
		'bollinger_percent_b__40': calculate_percent_b(df.close, n=40),
		'bollinger_percent_b__80': calculate_percent_b(df.close, n=80),
		'bollinger_percent_b__150': calculate_percent_b(df.close, n=150),

		# 'noise_1': pd.Series(np.random.random(df.shape[0]), index=df.index),
		# 'noise_2': pd.Series(np.random.random(df.shape[0]), index=df.index),
		# 'noise_3': pd.Series(np.random.random(df.shape[0]), index=df.index),
		# 'noise_4': pd.Series(np.random.random(df.shape[0]), index=df.index),
		# 'noise_5': pd.Series(np.random.random(df.shape[0]), index=df.index),

	}, index=df.index)

	_X = _X.loc[t0]

	# _df = pd.concat([_X, _y.to_frame()])
	_df = _X.copy()
	_df['y'] = _y
	_dfs.append(_df)

training_data = pd.concat(_dfs, axis=0)
training_data = training_data.dropna(how='any', axis=0)
training_data = training_data.dropna(how='all', axis=1)


classifier = tree.DecisionTreeClassifier(max_depth=7)

# TODO: Use a less complex model
# classifier = ensemble.RandomForestClassifier(
# 	max_depth=3,
# 	n_estimators=100,
# 	# min_weight_fraction_leaf=0.0001,
# 	# min_impurity_decrease=0.0001,
# 	verbose=1,
# 	oob_score=True,
# )

y = training_data['y']
X = training_data.drop(columns=['y'])
classifier.fit(X, y)
y_hat = classifier.predict(X)
y_hat = pd.Series(y_hat, name='y_hat', index=y.index)
print(f'Accuracy: {100 * (y == y_hat).mean():.2f}%')
# print(classifier.oob_score_)





##################################



# pieces_of_X = []
# pieces_of_y = []
# symbols = [...]
# for symbol in symbols:
# 	df = load_eod(symbol)

# 	# Trade the stock once every 10 trading days
# 	t0: pd.Index = df.index[::10]

# 	# Exit after 8 calendar days
# 	t1 = t0 + pd.Timedelta(days=8)

# 	# A +1 if the stock goes up and a 0 if it goes down
# 	# This has length equal to t0 and t1
# 	piece_of_y = (df.close[t1] > df.close[t0]).astype('int64')

# 	# This has a length equal to df.shape[0]
# 	piece_of_X = do_a_ton_of_technical_indicators(df)

# 	# Now it has length equal to t0, t1, and piece_of_y
# 	piece_of_X = piece_of_X[t0]

# 	pieces_of_X.append(piece_of_X)
# 	pieces_of_y.append(piece_of_y)

# X = pd.concat(pieces_of_X, axis=0)
# y = pd.concat(pieces_of_y, axis=0)

