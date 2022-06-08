#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:18:24 2022

This script leverages luigi to run grid search optimization on 6,305 ticker dataframe. 

Pending: 
    * currently, running one full time through the parameter grid 
    * I thought I fixed these issues already, but this is still running into issues 
    such as assertion error: spent more cash than you have and nans in volatility, 
    -infinity in log_max_drawdown_ratio, and 0.0 in final_equity. First, check to 
    see if I have my files in order. 
    * time the process 
    * turn parameter grid into json file for consistency 

@author: ellenyu
"""
import luigi
import json
import pandas as pd
import datetime
from vqti.cci_ellen import * 
from vqti.signal_ellen import * 
from vqti.performance import calculate_cagr, calculate_sharpe_ratio
from pypm import metrics, signals, data_io, simulation 

class returnPrices(luigi.Task):
    url = luigi.parameter.Parameter(default = 'https://s3.us-west-2.amazonaws.com/public.box.conlan.io/e67682d4-6e66-48f8-800c-467d2683582c/0b40958a-fa6f-448f-acbf-9d5478308cf5/prices.csv')
    
    def run(self):    
        # read in the data 
        df = pd.read_csv(self.url, parse_dates = ['date'], index_col ='date')
        
        # transform the data
        price_df = df[['ticker', 'close_split_adjusted']]\
                    .pivot_table(index = 'date', columns = 'ticker', 
                                 values = 'close_split_adjusted')
         
        # write to disk
        ## convert dataframe to json string
        price_json = price_df.to_json(orient = "columns")
        #convert json string to dictionary
        parsed = json.loads(price_json)
        # save dictionary to json file
        with self.output().open('w') as file_out: 
              json.dump(parsed, file_out)
              
    def output (self): 
        return luigi.LocalTarget('luigi_files/price_df.json')



class runSimulation(luigi.Task):
    # Note, url is not used in this task, but still have to pass it through
    url = luigi.parameter.Parameter(default = 'https://s3.us-west-2.amazonaws.com/public.box.conlan.io/e67682d4-6e66-48f8-800c-467d2683582c/0b40958a-fa6f-448f-acbf-9d5478308cf5/prices.csv')
    path_to_grid = luigi.parameter.Parameter(default = 'luigi_files/parameter_grid.txt')

    def requires(self):
        # read json in as dataframe - Pending
        return returnPrices(url = self.url)

    def run(self):
        with open(self.path_to_grid, 'r') as file_in:
            
            for line in file_in:
                
                # read in parameters   
                window = json.loads(line).get('window_length')
                strategy_type = json.loads(line).get('strategy_type')
                upper_band = json.loads(line).get('upper_band')
                lower_band = json.loads(line).get('lower_band')
                max_positions = json.loads(line).get('max_active_positions')
                                    
                # run simulation on set of parameters
                with self.input().open('r') as file_in, self.output().open('w') as file_out:
                    
                    # load in prices (dictionary)
                    price_dict = json.load(file_in)

                    # calculate indicator    
                    # # working directly with dictionary data structure
                    # cci_dict={}                                                            
                    # for k,v in price_dict.items():          
                    #     #file_out.write(str(type(k))+'\n') #<class 'str'>
                    #     #file_out.write(str(type(v))+'\n') #<class 'dict'>
                    #     #file_out.write(str(len(v.values()))+'\n') #5969
                    #     # cast NoneType in dictionary to python float('nan') because of 
                    #     # TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
                    #     v = {k: float('nan') if not v else v for k, v in v.items()} 
                    #     cci_dict[k] = python_ccimodified_loop(list(v.values()), window=window)
                    
                    ## converting dict to dataframe 
                    price_df = pd.DataFrame(price_dict)
                    #price_df = pd.DataFrame.from_dict(price_dict) #??? Perhaps from_dict() implementation is hard as opposed to squishy
                    # file_out.write(str(price_df.index) + '\n') # Index(['883526400000', '883699200000', etc. 
                    # file_out.write(str(type(price_df.index[0])) + '\n') # <class 'str'>
                    #convert json epoch str index to pandas datetime index 
                    price_df.index = pd.to_datetime(price_df.index, unit='ms')
                    
                    # set nans in price_df to 0
                    price_df = price_df.fillna(0)
                    
                    # calculate cci
                    cci_df = price_df.apply(lambda col: python_ccimodified_loop(col.tolist(), 
                                                                                    window = window), axis = 0) 
                    # translate to signals 
                    signal_df = cci_signals_generator(series = cci_df, strategy_type = strategy_type, 
                                                      upper_band = upper_band, lower_band = lower_band)
                    
                    # do nothing on the last day
                    signal_df.iloc[-1] = 0
                    
                    # define the preference matrix 
                    preference = price_df.apply(metrics.calculate_rolling_sharpe_ratio, axis=0)
                    
                    # run the simulator
                    ## set up instance 
                    simulator = simulation.SimpleSimulator(
                        initial_cash=1_000_000,
                        max_active_positions=max_positions,
                        percent_slippage=0.0005,
                        trade_fee=1,
                    )
                    
                    ## call simulate 
                    simulator.simulate(price = price_df, signal = signal_df, preference = preference)
                    
                    ## call portfolio_history
                    portfolio_history = simulator.portfolio_history
                    
                    performance_metrics = {
                        
                    # Performance metrics
                    'percent_return': portfolio_history.percent_return,
                    'spy_percent_return': portfolio_history.spy_percent_return,
                    'cagr': portfolio_history.cagr,
                    'volatility': portfolio_history.volatility,
                    'sharpe_ratio': portfolio_history.sharpe_ratio,
                    'spy_cagr': portfolio_history.spy_cagr,
                    'excess_cagr': portfolio_history.excess_cagr,
                    'jensens_alpha': portfolio_history.jensens_alpha,
                    'dollar_max_drawdown': portfolio_history.dollar_max_drawdown,
                    'percent_max_drawdown': portfolio_history.percent_max_drawdown,
                    'log_max_drawdown_ratio': portfolio_history.log_max_drawdown_ratio,
                    'number_of_trades': portfolio_history.number_of_trades,
                    'average_active_trades': portfolio_history.average_active_trades,
                    'final_equity': portfolio_history.final_equity,
                    
                    # Grid search parameters
                    'window_length': window,
                    'strategy_type': strategy_type,
                    'lower_band': lower_band, 
                    'upper_band': upper_band,
                    'max_active_positions': max_positions,
                    
                    }
                    
                    #write performance metric dictionary to file 
                    file_out.write(json.dumps(performance_metrics))
                    
                    # Verifying while I build 
                    ## write parameters to disk - verified 
                    # file_out.write(str(window)+'\n')
                
                    ## write price_dict to disk - verified 
                    # file_out.write(str(type(price_dict))+'\n')
                    # json.dump(price_dict, file_out)
                 
                    ## write cci_dict to disk - verified  
                    # json.dump(cci_dict, file_out)
                    
                    ## write price_df, cci_df, and signal_df to disk - verified 
                    ### convert dataframe to json string
                    # price_json = price_df.to_json(orient = "columns")
                    ### convert json string to dictionary - double check if I should put this step here or later... Return to this
                    # parsed = json.loads(price_json)
                    ### save dictionary to json file
                    ## file_out.write(str(cci_df.isnull().sum().sort_values())+'\n')
                    # json.dump(parsed, file_out)
                       
    def output(self):
        return luigi.LocalTarget('luigi_files/optimization.json')

if __name__ == '__main__':
    luigi.run()