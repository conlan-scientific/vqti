#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:59:45 2022

This script generates a first model 

@author: ellenyu
"""
from vqti.load import load_eod
from vqti.indicators.cci_signal import *
from pypm.signals import *


# cull together tehcnical indicator code I want to use 
def do_a_ton_of_technical_indicators(series: pd.Series, strategy_type: str= 'reversal', upper_band: int=3, lower_band: int=-3, window: int=20, factor: int=1)  -> pd.DataFrame:
    #cci
    cci = create_cci_signal_panda(df.close, strategy_type=strategy_type, upper_band=upper_band, lower_band=lower_band, window=window, factor=factor)
    
    #bollinger 
    bollinger = create_bollinger_band_signal(df.close)
    
    #hullma
    
    #macd
    macd = create_macd_signal (df.close) 
    #aroon
    
    #rsi
    
    df=pd.concat([cci,bollinger,macd],axis=0)

# Chris' code for generating training data 
pieces_of_X = []
pieces_of_y = []
symbols = [...]
for symbol in symbols:
	df = load_eod(symbol)

	# Trade the stock once every 10 trading days
	t0: pd.Index = df.index[::10]

	# Exit after 8 calendar days
	t1 = t0 + pd.Timedelta(days=8)

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

# Chris's code for machine learning 
# from sklearn import svm
# classifier = svm.SVC(gamma=0.001, C=100)

from sklearn import tree
classifier = tree.DecisionTreeClassifier(max_depth=6)

classifier.fit(X, y)
y_hat = classifier.predict(X)
y_hat = pd.Series(y_hat, name='y_hat')
print(f'Accuracy: {100 * (y == y_hat).mean():.2f}%')


# knn 

# this bayesian technique