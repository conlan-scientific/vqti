from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

from vqti.performance import (
	calculate_cagr,
	calculate_annualized_volatility,
	calculate_sharpe_ratio,
)

from vqti.indicators import calculate_percent_b


if __name__ == '__main__':

	df = load_eod('AWU')






