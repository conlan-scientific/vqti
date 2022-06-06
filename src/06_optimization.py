from itertools import product


def prepare_grid_from_kwargs(kwargs):
	"""
	Wizardry to convert this ...
	{
		'a': [1,2,3,4], 
		'b': [5,6,7,8]
	}

	To this ...
	[
		{'a': 1, 'b': 5},
		{'a': 1, 'b': 6},
		{'a': 1, 'b': 7},
		{'a': 1, 'b': 8},
		{'a': 2, 'b': 5},
		{'a': 2, 'b': 6},
		{'a': 2, 'b': 7},
		{'a': 2, 'b': 8},
		{'a': 3, 'b': 5},
		{'a': 3, 'b': 6},
		{'a': 3, 'b': 7},
		{'a': 3, 'b': 8},
		{'a': 4, 'b': 5},
		{'a': 4, 'b': 6},
		{'a': 4, 'b': 7},
		{'a': 4, 'b': 8}
	]

	Which is pretty useful for creating lists of parameters for a grid search. 
	"""
	assert all([isinstance(v, list) for v in kwargs.values()]), \
		'All kwargs values must be lists.'
	kwargs_list = [[(k, v) for v in v_list] for k, v_list in kwargs.items()]
	kwargs_list = [dict(d) for d in product(*kwargs_list)]	
	return kwargs_list





