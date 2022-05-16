
import math
from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot
import pandas as pd
import numpy as np
from typing import List


"""
Bollinger Bands, %B, and Bandwidth
"""

# TODO: Write a pure python function for the SMA
# TODO: Write a pure python function for the rolling standard deviation


def python_pB (values: List[float], n: int = 4):
    # TODO: Don't load data inside function
    # TODO: The input should be a list of floats

    values = load_eod('AWU') # loads the dataframe in question
    value_range = values.loc[:,'close'] # loads only the values in a specific column. In this case, "close".

    #High Level Variables
    N = 1 #A multiplier for the standard deviation
    
    amov = 0 #Initializes the moving average
    
    dev = 0 #Initializes the differences from the moving average
    devsum = 0 #Initializes the sum of the differences

    var = 0 #Initializes the variance
    
    stddev_j = [] #Initializes the standard deviation list
    
    amovl = []
    rolling_total = 0
    
    BBu = 0 #The initialization of the Upper Bollinger Band values
    BBl = 0 #The initialization of the Lower Bollinger Band values
    pB = [] #Initializes the %B list    
    bandwidth = []

    #Simple moving average initialization
    rolling_total = sum(value_range[:n]) #Adds up the initial numbers associated with the moving average range
    amov = rolling_total/n #Calculates the first moving average
    amovl.append(amov) #For testing that amov is working correctly. It should update with new values of amov
    
    
    #Standard deviation initialization   
    for a in range (0, n):
        dev = value_range[a] - amov #The initial interval-summed differences between the column values and the moving average
    
    devsum = 0 + dev
    
    var = devsum**2 #The variance of the deviations
    stddev = math.sqrt(var) #The initial standard deviation 
    stddev_j.append(N*stddev) #Defines the number of standard deviations under consideration and places each incremented value in a list
    
    for k in range (n, len(value_range),n): #Loops the moving average and standard deviation formulas to form a standard deviation list
        rolling_total = rolling_total - value_range[k-n] + value_range[k] #Subtracts the oldest value off of the moving average total, then adds the newest value onto the moving average total
        amov = rolling_total/n #Calculates the moving average based on the current rolling total
        amovl.append(amov) #Appends amovl with the newest value of amov
        
        dev = value_range[k] - amov #The listed differences between the close column values and the moving average at each interval as defined in the loop
     

        devsum = 0 + dev
        
        var = devsum**2 #The variance of the differences
        stddev = math.sqrt(var) #The standard deviation
        stddev_j.append(N*stddev) #Adds the standard deviation onto the end of the standard deviation list

        BBu = amov + stddev #Adds the moving average and the standard deviation at each increment and places the final value at the end of the Upper Bollinger Band list
        BBl = amov - stddev #Subtracts the standard deviation from the moving average at each increment and places the final value at the end of the Lower Bollinger Band list 
        
        pB.append((value_range[k] - BBl)/(BBu - BBl))
    
        bandwidth.append((BBu-BBl)/amov)
        
    return amovl
    
    



if __name__=="__main__":    
   
    df = load_eod('AWU')
    c = df.loc[:,'close'] 
    d = c[0:len(c)]
    print (d)
    
    
    amov_result = python_pB(df)
    print (amov_result)

    