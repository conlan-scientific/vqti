import pandas as pd
from vqti.load import load_eod
from pypm.data_io import get_all_symbols
from vqti.indicators.hma import calculate_numpy_matrix_hma
from vqti.indicators.hma_signals import calculate_hma_zscore
from sklearn import tree
from sklearn import ensemble
import numpy as np
from hull_data import calculate_tbm_labels

symbols = get_all_symbols() # ['AWU', 'BGH', ...]

_dfs = list()
for symbol in symbols[:5]:
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
	event_labels, event_spans = calculate_tbm_labels(price_series, event_index)
	_y = pd.Series(_y, index=t0, name='y')
	t0 = t0[_y != 0]
	t1 = t1[_y != 0]


	_X = pd.DataFrame({
		'hma_trend_10': calculate_numpy_matrix_hma(df.close, 16),
		'hma_trend_25': calculate_numpy_matrix_hma(df.close, 25),
		'hma_trend_49': calculate_numpy_matrix_hma(df.close, 49),
		'hma_trend_81': calculate_numpy_matrix_hma(df.close, 81),

        'hma_zscore_25_49': calculate_hma_zscore(df.close,25,49),
        

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

