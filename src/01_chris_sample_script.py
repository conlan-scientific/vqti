from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd

@time_this
def get_the_cumsum(close: pd.Series) -> pd.Series:
	return close.cumsum()

if __name__ == '__main__':
	
	df = load_eod('AWU')
	print(df)

	# df.close.plot()
	# plt.show()

	result = get_the_cumsum(df.close)


