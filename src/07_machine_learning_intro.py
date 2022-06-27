
from vqti.load import load_eod
df = load_eod('AWU')

"""

How to do machine learning on time series data

The fundamental problem of machine learning
-------------------------------------------
Given a vector y ...
Build a matrix X ...
Determine f to minimize epsilon. 

y = f(X) + epsilon^2

y can be called the target, the label, the objective ...

y is *a trade* typically in the stock market.

signal matrix is typically a bunch of +1's, 0's, and -1's

Per scikit-learn ...
signal_df = y_hat = my_model.predict(X)

What does a row of X represent?
[[..., ..., ..., ...],
 [..., ..., ..., ...],
 [..., ..., ..., ...],
 [..., ..., ..., ...],
 ...
 [..., ..., ..., ...],]


An element of y, y_i, represents a trade that starts on t0 and ends on t1, then
a row of X, X_i, can be any information we had access to on t0.

All of the t0's are the *event starts* or *trade starts*. All t1's are 
*event ends* or *trade ends*. 
"""

pieces_of_X = []
pieces_of_y = []
symbols = [...]
for symbol in symbols:
	df = load_eod(symbol)

	# t0 are the events
	# Trade the stock once every 10 trading days
	t0: pd.Index = df.index[::10]

	# t1 are trade ends
	# Exit after 8 calendar days
	t1 = t0 + pd.Timedelta(days=8)

	# The label, y
	# A +1 if the stock goes up and a 0 if it goes down
	# This has length equal to t0 and t1
	piece_of_y = (df.close[t1] > df.close[t0]).astype('int64')

	# This has a length equal to df.shape[0]
	piece_of_X = do_a_ton_of_technical_indicators(df)

	# Now it has length equal to t0, t1, and piece_of_y
	piece_of_X = piece_of_X[t0]

	pieces_of_X.append(piece_of_X)
	pieces_of_y.append(piece_of_y)

X = pd.concat(pieces_of_X, axis=0)
y = pd.concat(pieces_of_y, axis=0)

# Now just do normal machine learning ...

# Out of the box with sklearn...
# model.fit(X, y)
# model.predict(X)









