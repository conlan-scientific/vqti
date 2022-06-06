#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:34:47 2022

Previously, I've tried joblib successfully and this time, I've tried running two python scripts at once. 
This script tries to implement luigi to handle our optimization - this is a work in progress version. 

Pending: 
    There is something wrong with writing prices dataframe in json to disk so, have to fix that 
    before moving forward in runSimulation task 
 
@author: ellenyu
"""
import luigi
import json
import pandas as pd
from cci_ellen import * 

# Loaded these in too, but not 100% need all of it. Return to this 
from signal_ellen import * 
from simulate_ellen import *
from vqti.performance import calculate_cagr, calculate_sharpe_ratio
from pypm import metrics, signals, data_io, simulation 

class returnPrices(luigi.Task):
    url = luigi.parameter.Parameter(default='https://s3.us-west-2.amazonaws.com/public.box.conlan.io/e67682d4-6e66-48f8-800c-467d2683582c/0b40958a-fa6f-448f-acbf-9d5478308cf5/prices.csv')
    
    def run(self):    
       # read in the data 
       df = pd.read_csv(self.url, parse_dates = ['date'], index_col='date')

       # transform the data
       price_df = df[['ticker', 'close_split_adjusted']]\
                .pivot_table(index = 'date', columns='ticker', values='close_split_adjusted')
                
       # write to disk
       with self.output().open('w') as file_out: 
           json.dumps(price_df)
        
    def output (self): 
        return luigi.LocalTarget('luigi_files/prices_df.json')

class runSimulation(luigi.Task):
    # Note, url is not used in this task, but still have to pass it through
    url = luigi.parameter.Parameter(default='https://s3.us-west-2.amazonaws.com/public.box.conlan.io/e67682d4-6e66-48f8-800c-467d2683582c/0b40958a-fa6f-448f-acbf-9d5478308cf5/prices.csv')
    path = luigi.parameter.Parameter(default='parameter_grid.txt')

    def requires(self):
        # read json in as dataframe - Pending
        return json.loads(returnPrices(url=self.url))

    def run(self):
        with open(self.path, 'r') as file_in:
            for line in file_in:
                    # read in parameters   
                    window = json.loads(line).get('window_length')
                    strategy_type = json.loads(line).get('strategy_type')
                    upper_band = json.loads(line).get('upper_band')
                    lower_band = json.loads(line).get('lower_band')
                    max_position = json.loads(line).get('max_active_position')
                    
                    ## write parameters to disk (this is a test)
                    #file_out.write(str(window)+'\n')
                    
                    # write out price_df (this is a test)
                    with self.output().open('w') as file_out:
                        json.dumps(self.input()) 
                    # calculate indicator 
                    #cci_df = price_df.apply(lambda col: python_ccimodified_loop(col.tolist(), 
                                                                                      #window=window), axis=0)
                    # # translate to signals 
                    # signal_df = cci_signals_generator(series=indicator_df, strategy_type=strategy_type, 
                    #                                   upper_band=upper_band, lower_band=lower_band)

                                   
    def output(self):
        ## write parameters to disk (this is a test)
        #return luigi.LocalTarget('luigi_files/test.txt')
        
        return luigi.LocalTarget('luigi_files/price_df_write.json')