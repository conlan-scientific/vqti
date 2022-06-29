import os
import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.preprocessing import StandardScaler
from hull_load_data import (load_data, 
                       load_volume_data, 
                       calculate_hull_features,
                       load_volume_and_revenue_data)

from pypm.ml_model.events import calculate_events
from pypm.ml_model.labels import calculate_labels
from pypm.ml_model.features import calculate_features
from pypm.ml_model.model import calculate_model
from pypm.ml_model.weights import calculate_weights
from hull_volume_filter import calculate_volume_pct_change_events
from sklearn import ensemble
from sklearn import tree

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = '\\'.join(os.path.dirname(__file__).split("/"))
VQTI_DIR = os.path.dirname(SRC_DIR)


if __name__ == '__main__':

    # All the data we have to work with
    symbols, eod_data, volume_data, revenue_data = load_volume_and_revenue_data()
    
    # The ML dataframe for each symbol, to be combined later
    df_by_symbol: Dict[str, pd.DataFrame] = dict()

    # Build ML dataframe for each symbol
    for symbol in symbols:

        # Get volume, revenue and price series
        volume_series = volume_data[symbol].dropna()
        revenue_series = revenue_data[symbol].dropna()
        price_series = eod_data[symbol].dropna()
        price_index = price_series.index
        
        # Section 7.1 (Events)
        # Get events, labels, weights, and features
        event_index = calculate_volume_pct_change_events(volume_series)

        # Section 7.2 (Triple-barrier method)
        # Calculating y (but also t1)
        event_labels, event_spans = calculate_labels(price_series, event_index)

        # Section 7.3 (Weights)
        # 1   1.1
        # 0   0.5
        # 0   0.7
        # .   0.6
        # .   0.3 
        # .   1.3
        # 1   1.1
        # 0   1.7

        # x                                     o
        #         x                             o
        #                    x                  o
        #                                              x      o
        # ------------------------------------------------------ t ---> 
        weights = calculate_weights(event_spans, price_index)


        features_df = calculate_hull_features(price_series, volume_series, revenue_series)

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
    # classifier = tree.DecisionTreeClassifier(max_depth=7)
    # classifier = ensemble.RandomForestClassifier(
    #     max_depth=3,
    #     n_estimators=100,
    #     # min_weight_fraction_leaf=0.0001,
    #     # min_impurity_decrease=0.0001,
    #     verbose=1,
    #     oob_score=True,
    # )
    
    # y = df['y']
    # X = df.drop(columns=['y'])
    # classifier.fit(X, y)
    # y_hat = classifier.predict(X)
    # y_hat = pd.Series(y_hat, name='y_hat', index=y.index)
    # print(f'Accuracy: {100 * (y == y_hat).mean():.2f}%')
    # print(classifier.oob_score_)

    # Save the model
    # dump(classifier, os.path.join(SRC_DIR, 'ml_model.joblib'))