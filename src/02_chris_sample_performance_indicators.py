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

	percent_b = calculate_percent_b(df.close)
	df = df.assign(percent_b=percent_b)































"""

P_t   (Portfolio)
C_t   (Cash)
E_t   (Equity)

E_t = P_t + C_t
P_t = sum of all shares times prices

Measure performance on E_t

-------

Risk-adjusted return

1. Measure return
Overall return = (E_T / E_0) - 1
^^^ Annualize it, and you have the CAGR

2. Measure risk
Standard deviation of return series of the equity curve
sigma_{r_t}
r_t = (E_t / E_{t-1}) - 1
^^^ Annualize it, and you have "Volatility"

3. Divide one by the other
Sharpe ratio is the CAGR over the volatility.

"""