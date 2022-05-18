# LONG ONLY

# This is a dataframe where each column is stock symbol
# The index is a date-ascending date index
# A value represents the closing price
prices_df = load_all_of_my_data()

#             AWU   BDF   ERG
# 2020-01-01  123   423   543
# 2020-01-02  124   426   542
# 2020-01-03  ...   ...   ...
# ... 

# This is a dataframe with the same structure, 
# but it is filled with only -1's, 0's, and 1's
signal_df = calculate_the_signals(prices_df)

#             AWU   BDF   ERG
# 2020-01-01    0     1     0
# 2020-01-02   -1     0     0
# 2020-01-03  ...   ...   ...
# ... 

assert prices_df.index.equals(signal_df.index)

max_assets = 20
dt_index = prices_df.index
cash = starting_cash = 100_000
portfolio: List[str] = list()
equity_curve = dict()


# Pretend you are walking through time trading over the course of ten years
for date in dt_index:

	signals = signal_df.loc[date] 

	stocks_im_going_to_buy: List[str] = signals[signals == 1].index.tolist()
	stocks_im_going_to_sell: List[str] = signals[signals == -1].index.tolist()
	stocks_im_holding: List[str] # This is determined by which stocks you bought previously

	# Mess with your cash and portfolio to sell stocks
	for stock in stocks_im_going_to_sell:
		cash += ...
		portfolio.pop(stock)
		stocks_im_holding.pop(stock)

	# What do I do with too many buy signals?
	# Mess with your cash and portfolio to buy stocks
	for stock in stocks_im_going_to_buy:
		# This is the "compound your gains and losses" approach to cash management
		# Also called the "fixed number of slots" approach
		cash_to_spend = cash / (max_assets - len(stocks_im_holding))

		# Straight from the book ... See page 68
		# self.free_position_slots = self.max_active_position - self.active_positions_count
		# cash_to_spend = self.cash / self.free_position_slots

		cash -= cash_to_spend
		portfolio.append(stock)
		stocks_im_holding.append(stock)

	equity_curve[date] = cash + # sum of my portfolio value


# Plot the equity curve
# Measure the sharpe ratio
# You're done.

