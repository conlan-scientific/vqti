import pandas as pd
from hullma import numpy_matrix_hma
from vqti.load import load_eod
from pypm.data_io import get_all_symbols
from vqti.indicators.hma import calculate_numpy_matrix_hma

symbols = get_all_symbols() # ['AWU', 'BGH', ...]

_dfs = list()
for symbol in symbols[:3]:
	print('Preparing training data for', symbol, '...')
	df = load_eod(symbol)

	# Trade the stock once every 10 trading days
	t0: pd.Index = df.index[::10]	

	# Exit after 20 trading days
	t1 = pd.Index(t0[2:].tolist() + [t0[-1], t0[-1]])

	# A +1 if the stock goes up and a 0 if it goes down
	# This has length equal to t0 and t1
	assert t0.shape == t1.shape
	_y = (df.close[t1].values > df.close[t0].values).astype('int64')
	_y = pd.Series(_y, index=t0, name='y')

	_X = pd.DataFrame({
		'hma_trend_10': numpy_matrix_hma(df.close, 16),
		'hma_trend_10': numpy_matrix_hma(df.close, 25),
		'hma_trend_10': numpy_matrix_hma(df.close, 49),
		'hma_trend_10': numpy_matrix_hma(df.close, 81),
	}, index=df.index)
	_X = _X.loc[t0]
	# _df = pd.concat([_X, _y.to_frame()])
	_df = _X.copy()
	_df['y'] = _y
	_dfs.append(_df)

training_data = pd.concat(_dfs, axis=0)
training_data = training_data.dropna(how='any', axis=0)
training_data = training_data.dropna(how='all', axis=1)


from sklearn import tree
from sklearn import ensemble
classifier = tree.DecisionTreeClassifier()
classifier = ensemble.RandomForestClassifier(
	max_depth=5,
	n_estimators=100,
	# min_weight_fraction_leaf=0.0001,
	# min_impurity_decrease=0.0001,
	verbose=1,
	oob_score=True,
)

y = training_data['y']
X = training_data.drop(columns=['y'])
classifier.fit(X, y)
y_hat = classifier.predict(X)
y_hat = pd.Series(y_hat, name='y_hat', index=y.index)
print(f'Accuracy: {100 * (y == y_hat).mean():.2f}%')
print(f'Out of Bag Score: {classifier.oob_score_}')
