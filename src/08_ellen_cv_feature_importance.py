#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jun 27 12:24:36 2022

The challenge - use the cross validator on dataset and have some conclusions. 

Note, used revenue data instead of alternative revenue data.

@author: ellenyu

"""

import os
import pandas as pd
import numpy as np
from typing import Dict

from joblib import dump

from pypm.ml_model.data_io import load_data
from pypm.ml_model.events import calculate_events
from pypm.ml_model.labels import calculate_labels
from pypm.ml_model.features import calculate_features
from pypm.ml_model.model import calculate_model
from pypm.ml_model.weights import calculate_weights

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    # All the data we have to work with
    symbols, eod_data, rev_data= load_data()

    # The ML dataframe for each symbol, to be combined later
    df_by_symbol: Dict[str, pd.DataFrame] = dict()

    # Build ML dataframe for each symbol
    for symbol in symbols:

        # Get revenue and price series
        #revenue_series = alt_data[symbol].dropna()
        revenue_series = rev_data[symbol].dropna()
        price_series = eod_data[symbol].dropna()
        price_index = price_series.index

        # Get events, labels, weights, and features
        event_index = calculate_events(revenue_series)
        event_labels, event_spans = calculate_labels(price_series, event_index)
        weights = calculate_weights(event_spans, price_index)
        features_df = calculate_features(price_series, revenue_series)

        # Subset features by event dates
        features_on_events = features_df.loc[event_index]

        # Convert labels and events to a dataframe
        labels_df = pd.DataFrame(event_labels)
        labels_df.columns = ['y']

        # Converts weights to a dataframe
        weights_df = pd.DataFrame(weights)
        weights_df.columns = ['weights']

        # Concatenate features to labels
        df = pd.concat([features_on_events, weights_df, labels_df], axis=1)
        df_by_symbol[symbol] = df

    # Create final ML dataframe
    df = pd.concat(df_by_symbol.values(), axis=0)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    print(df)

    # Fit the model
    classifier = calculate_model(df)

    # Save the model
    dump(classifier, os.path.join(SRC_DIR, 'ml_model.joblib'))

## First run 
# Feature importances
# 30_day_return            0.096
# 7_day_return             0.096
# 30_day_vol               0.070
# 90_day_return            0.068
# 180_day_return           0.066
# 360_day_vol              0.065
# 360_day_revenue_delta    0.065
# 360_day_return           0.063
# 7_day_vol                0.062
# 180_day_vol              0.061
# 180_day_revenue_delta    0.060
# 90_day_revenue_delta     0.058
# 90_day_vol               0.058
# 7_day_revenue_delta      0.057
# 30_day_revenue_delta     0.056

# Cross validation scores
# [52.  52.  52.6 45.2 52.4 54.2 49.1 48.  52.4 49.9 48.  53.1 56.7 50.6
#  52.2 49.9 56.1 51.3 51.3 53.6]

# Baseline accuracy 42.2%
# OOS accuracy 51.5% +/- 5.3%
# Improvement 4.0 to 14.7%


## Second run 
# Feature importances
# 30_day_return            0.098
# 7_day_return             0.094
# 30_day_vol               0.070
# 360_day_vol              0.067
# 90_day_return            0.066
# 360_day_revenue_delta    0.066
# 7_day_vol                0.065
# 360_day_return           0.063
# 180_day_return           0.062
# 180_day_revenue_delta    0.062
# 90_day_vol               0.059
# 90_day_revenue_delta     0.059
# 7_day_revenue_delta      0.059
# 180_day_vol              0.057
# 30_day_revenue_delta     0.055

# Cross validation scores
# [52.3 53.7 48.2 51.5 53.7 53.7 51.3 52.3 52.6 48.9 56.6 49.1 51.3 51.7
#  52.7 45.9 52.7 52.6 53.4 49.5]

# Baseline accuracy 42.2%
# OOS accuracy 51.7% +/- 4.7%
# Improvement 4.8 to 14.2%

## Quick observations 
# Note, we used revenue data instead of alternative revenue data 
# Similar to what's discussed in the book, what's important are: 
    # near term return 
    # near term volatility 
    # in terms of revenue delta, the YoY revenue change is most important
 





"""
RandomForestClassifier() ...

We're going to make n_estimators=1000 separate decision trees.

Give me two-thirds of the rows. Just toss out the rest.

                    + 
                   / \   <--- Come up with a rule 
                  +   +

    To come up with a rule ...
        Select a random subset of two-thirds of the columns.
        Loop through the columns randomly.
        Test breakpoints somewhat random.
        Keep the rule with the highest classification accuracy.

                x4 >= 234.5    <---- I found my rule!

                    + 
                   / \
                  +   +
                 / \      <---- Now come up with this rule
                +   +

Finish the tree.

Now do it 1000 times over for each tree.






















"""



















