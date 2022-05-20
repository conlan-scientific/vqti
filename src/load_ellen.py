"""
Created on Thu May 19, 2022

@author: ellenyu

Pending: 
    * Address the questions and work arounds I've denoted below
    
"""
import os
import pandas as pd
import glob
from cci_ellen import *
from vqti.profile import time_this 

MODULE_DIR = os.path.dirname('/Users/ellenyu/vqti/src/vqti/load_all_ellen.py')
#MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) # Path where current file is located 
SRC_DIR = os.path.dirname(MODULE_DIR) # Directory aka parent folder of path
GIT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(GIT_DIR, 'data')
EOD_DIR = os.path.join(DATA_DIR, 'eod')
REVENUE_DIR = os.path.join(DATA_DIR, 'revenue')

# print(os.path.abspath(__file__)) # On my computer, /Users/ellenyu/vqti/src/vqti/load_all_ellen.py
# print(MODULE_DIR) 
# print(SRC_DIR)
# print(GIT_DIR)
# print(DATA_DIR)
# print(EOD_DIR)
# print(REVENUE_DIR)

def load_eod(symbol: str) -> pd.DataFrame:
    filepath = os.path.join(EOD_DIR, f'{symbol}.csv')
    #print(filepath)
    return pd.read_csv(
		filepath, 
		parse_dates=['date'], 
		index_col='date', 
		dtype='float64',
	)

def load_all(from_directory: str = EOD_DIR, extension: str = 'csv', to_directory: str = SRC_DIR) -> pd.DataFrame:
    # Change directory to directory that house the data files
    os.chdir(from_directory)
    # Collect all filenames of a particular extension into a list
    all_filenames: List[str] = [i for i in glob.glob('*.{}'.format(extension))]
    #print(all_filenames)
    
    list_of_dataframes = []
    
    for filename in all_filenames:
        df = pd.read_csv(filename, 
                         parse_dates=['date'], # ??? Read the documentation, but still not completely sure what parse_date adds to function
                         index_col='date', 
                         dtype='float64',
                         )
        df['ticker'] = filename.split('.')[0]
        list_of_dataframes.append(df)

    all_df = pd.concat(list_of_dataframes)
    
    assert all_df['ticker'].nunique() == len(all_filenames), "Tickers in dataframe does not equal number of files loaded in"
    
    # # If I want to export combined dataframe into csv
    # os.chdir(to_directory)
    # all_df.to_csv( "all_df.csv", index=False, encoding='utf-8-sig')
    
    return all_df 

# ??? How to run technical indicator by ticker on the ticker escapes me atm so, this is a temporary work around 
# ??? Do I need to specify window or **kwargs or don't need to specify
def load_all_with_cci(from_directory: str = EOD_DIR, extension: str = 'csv', to_directory: str = SRC_DIR, window: int = 20) -> pd.DataFrame:
    # Change directory to directory that house the data files
    os.chdir(from_directory)
    # Collect all filenames of a particular extension into a list
    all_filenames: List[str] = [i for i in glob.glob('*.{}'.format(extension))]
    #print(all_filenames)
    
    list_of_dataframes = []
    
    for filename in all_filenames:
        df = pd.read_csv(filename, 
                         parse_dates=['date'], # Read the documentation, but still not completely sure what parse_date adds to function ???
                         index_col='date', 
                         dtype='float64',
                         )
        #df['cci'] = pandas_series_cci_rolling(df.close)
        df['cci'] = pd.Series(python_cci_loop(df.close.tolist(), window=window), index=df.index)
        df['ticker'] = filename.split('.')[0]
        list_of_dataframes.append(df)

    all_df = pd.concat(list_of_dataframes)
    
    assert all_df['ticker'].nunique() == len(all_filenames), "Tickers in dataframe does not equal number of files loaded in"
    
    # # If I want to export combined dataframe into csv
    # os.chdir(to_directory)
    # all_df.to_csv( "all_df.csv", index=False, encoding='utf-8-sig')
    
    return all_df 

# ??? Say I want to exchange python_cci_loop with python_ccimodified_loop, how do I do that? The following is a temporary work around 
def load_all_with_ccimodified(from_directory: str = EOD_DIR, extension: str = 'csv', to_directory: str = SRC_DIR, window: int = 20) -> pd.DataFrame:
    # Change directory to directory that house the data files
    os.chdir(from_directory)
    # Collect all filenames of a particular extension into a list
    all_filenames: List[str] = [i for i in glob.glob('*.{}'.format(extension))]
    #print(all_filenames)
    
    list_of_dataframes = []
    
    for filename in all_filenames:
        df = pd.read_csv(filename, 
                         parse_dates=['date'], # Read the documentation, but still not completely sure what parse_date adds to function ???
                         index_col='date', 
                         dtype='float64',
                         )
        #df['cci'] = pandas_series_cci_rolling(df.close)
        df['cci'] = pd.Series(python_ccimodified_loop(df.close.tolist(), window=window), index=df.index)
        df['ticker'] = filename.split('.')[0]
        list_of_dataframes.append(df)

    all_df = pd.concat(list_of_dataframes)
    
    assert all_df['ticker'].nunique() == len(all_filenames), "Tickers in dataframe does not equal number of files loaded in"
    
    # # If I want to export combined dataframe into csv
    # os.chdir(to_directory)
    # all_df.to_csv( "all_df.csv", index=False, encoding='utf-8-sig')
    
    return all_df 

# After seeing sample simulator code, decided to write another load function [5-20-22]
@time_this
def load_all_onecolumn (from_directory: str = EOD_DIR, extension: str = 'csv', to_directory: str = SRC_DIR, column: str = 'close') -> pd.DataFrame:
    
    # Change directory to directory that house the data files
    os.chdir(from_directory)
    # Collect all filenames of a particular extension into a list
    all_filenames: List[str] = [i for i in glob.glob('*.{}'.format(extension))]
    #print(all_filenames)
    
    all_df = pd.DataFrame()

    for filename in all_filenames:
        df = pd.read_csv(filename, 
                         parse_dates=['date'], # ??? Read the documentation, but still not completely sure what parse_date adds to function
                         index_col='date', 
                         dtype='float64',
                         usecols=['date', column])
        
        # Set ticker name as the collumn name
        df.columns = [filename.split('.')[0]]
        
        # Concatenate each df to all_df
        all_df = pd.concat([all_df, df], axis=1)
    
    assert len(all_df.columns) == len(all_filenames), "Tickers in dataframe does not equal number of files loaded in"
    
    # # If I want to export combined dataframe into csv
    # os.chdir(to_directory)
    # all_df.to_csv( "all_df.csv", index=False, encoding='utf-8-sig')
    
    return all_df

@time_this
def load_all_onecolumn_with_cci (from_directory: str = EOD_DIR, extension: str = 'csv', to_directory: str = SRC_DIR, column: str = 'close', window: int = 20) -> pd.DataFrame:
    
    # Change directory to directory that house the data files
    os.chdir(from_directory)
    # Collect all filenames of a particular extension into a list
    all_filenames: List[str] = [i for i in glob.glob('*.{}'.format(extension))]
    #print(all_filenames)
    
    all_df = pd.DataFrame()

    for filename in all_filenames:
        df = pd.read_csv(filename, 
                         parse_dates=['date'], # ??? Read the documentation, but still not completely sure what parse_date adds to function
                         index_col='date', 
                         dtype='float64',
                         usecols=['date', column])
        # Calculate cci 
        cci = pd.Series(python_cci_loop(df[column].tolist(), window=window), index=df.index, name=filename.split('.')[0])
        
        # Concatenate each df to all_df
        all_df = pd.concat([all_df, cci], axis=1)
    
    assert len(all_df.columns) == len(all_filenames), "Tickers in dataframe does not equal number of files loaded in"
    
    # # If I want to export combined dataframe into csv
    # os.chdir(to_directory)
    # all_df.to_csv( "all_df.csv", index=False, encoding='utf-8-sig')
    
    return all_df

@time_this
def load_all_onecolumn_with_ccimodified (from_directory: str = EOD_DIR, extension: str = 'csv', to_directory: str = SRC_DIR, column: str = 'close', window: int = 20) -> pd.DataFrame:
    
    # Change directory to directory that house the data files
    os.chdir(from_directory)
    # Collect all filenames of a particular extension into a list
    all_filenames: List[str] = [i for i in glob.glob('*.{}'.format(extension))]
    #print(all_filenames)
    
    all_df = pd.DataFrame()

    for filename in all_filenames:
        df = pd.read_csv(filename, 
                         parse_dates=['date'], # ??? Read the documentation, but still not completely sure what parse_date adds to function
                         index_col='date', 
                         dtype='float64',
                         usecols=['date', column])
        # Calculate cci 
        cci = pd.Series(python_ccimodified_loop(df[column].tolist(), window=window), index=df.index, name=filename.split('.')[0])
        
        # Concatenate each df to all_df
        all_df = pd.concat([all_df, cci], axis=1)
    
    assert len(all_df.columns) == len(all_filenames), "Tickers in dataframe does not equal number of files loaded in"
    
    # # If I want to export combined dataframe into csv
    # os.chdir(to_directory)
    # all_df.to_csv( "all_df.csv", index=False, encoding='utf-8-sig')
    
    return all_df

    

if __name__ == '__main__':
 
    # df = load_eod("AWU")
    # print(df.shape)
    # print(df.head(20))
    
    # df_all = load_all()
    # print(df_all.shape)
    # print(df_all.head(20))
    
    # df_all_cci = load_all_with_cci()
    # print(df_all_cci.shape)
    # print(df_all_cci.head(20))
    
    # df_all_ccimodified = load_all_with_ccimodified()
    # print(df_all_ccimodified.shape)
    # print(df_all_ccimodified.head(20))
    
    # df_all_onecolumn = load_all_onecolumn()
    # print(df_all_onecolumn.shape)
    # print(df_all_onecolumn.head(20).iloc[:, :3])
    
    df_all_onecolumn_cci = load_all_onecolumn_with_cci(window=3)
    print(df_all_onecolumn_cci.shape)
    print(df_all_onecolumn_cci.head(20).iloc[:, :3])
    
    # df_all_onecolumn_ccimodified = load_all_onecolumn_with_ccimodified(window=5)
    # print(df_all_onecolumn_ccimodified.shape)
    # print(df_all_onecolumn_ccimodified.head(20).iloc[:, :3])