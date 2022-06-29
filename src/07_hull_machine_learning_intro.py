import pandas as pd
from  vqti.indicators.hma import calculate_numpy_matrix_hma
from vqti.load import load_eod
from pypm.data_io import get_all_symbols
from vqti.indicators.hma_signals import calculate_hma_zscore
import matplotlib.pyplot as plt
import numpy as np

symbols = get_all_symbols() # ['AWU', 'BGH', ...]

_dfs = list()
for symbol in symbols[:10]:
	print('Preparing training data for', symbol, '...')
	df = load_eod(symbol)
 
	# t0 are the events
	# Trade the stock once every 10 trading days
	t0: pd.Index = df.index[::10]	
	# Trade the stock once every 22 trading days (monthly)
	# t0: pd.Index = df.index[::22]
 
	#t1 are trade ends(event spans)
	# Exit after 20 trading days
	t1 = pd.Index(t0[2:].tolist() + [t0[-1], t0[-1]])
	# Exit after 365 calendar days
	# t1 = pd.Series(
	# 	df.index[[min(df.shape[0]-1, x) for x in df.index.searchsorted(t0 + pd.Timedelta(days=365))]], 
	# 	index=t0,
	# )


	# The event label, y
	# was the trade a winner or loser
	# A +1 if the stock goes up and a 0 if it goes down
	# This has length equal to t0 and t1
	assert t0.shape == t1.shape
	_y = (df.close[t1].values > df.close[t0].values).astype('int64')
	_y = pd.Series(_y, index=t0, name='y')
 
	
	# _y = [
	# 	(1 if x > 0.20 else (-1 if x < 0 else 0)) for x in \
	# 	(df.close[t1].values / df.close[t0].values - 1)
	# ]
	
	t0 = t0[_y != 0]
	t1 = t1[_y != 0]

	_X = pd.DataFrame({
		'hma_trend_16': calculate_numpy_matrix_hma(df.close, 16),
		'hma_trend_25': calculate_numpy_matrix_hma(df.close, 25),
		'hma_trend_49': calculate_numpy_matrix_hma(df.close, 49),
		'hma_trend_81': calculate_numpy_matrix_hma(df.close, 81),
		'hma_zscore_25_49': calculate_hma_zscore(df.close,25,49),
		# if you want to see how noise stacks up to your features.
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


from sklearn import tree
from sklearn import ensemble
# classifier = tree.DecisionTreeClassifier()
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

from sklearn.model_selection import ParameterGrid
grid = {'n_estimators': [200], 'max_depth':[3, 4,5], 'max_features': [1,2,3,4]}
# Determine the best hyperparameters to use and run the model to find accuracy
test_scores = []
# loop through the parameter grid, set hyperparameters, save the scores
for g in ParameterGrid(grid):
    classifier.set_params(**g) # ** is "unpacking" the dictionary
    classifier.fit(X, y)
    test_scores.append(classifier.score(X, y))  
# find best hyperparameters from the test score and print
best_idx = np.argmax(test_scores)
print(test_scores[best_idx])
print(ParameterGrid(grid)[best_idx])