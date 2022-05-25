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

# TODO: Write a pure python function for the rolling standard deviation


def python_sma (values: List[float], n) -> List[float]:
    #High Level Variables
    amovl = []
    rolling_total = 0

    #Simple moving average initialization
    rolling_total = sum(values[:n]) #Adds up the initial numbers associated with the moving average range
    amov = rolling_total/n #Calculates the first moving average
    amovl.append(amov) #For testing that amov is working correctly. It should update with new values of amov

    for k in range (n, len(values)): #Loops the moving average and standard deviation formulas to form a standard deviation list
        rolling_total = rolling_total - values[k-n] + values[k] #Subtracts the oldest value off of the moving average total, then adds the newest value onto the moving average total
        amov = rolling_total/n
        amovl.append(amov) #Appends amovl with the newest value of amov

    return amovl
    
def python_stddev (values, n) -> List [float]:
    #High Level Variables
    dev = 0
    var = 0
    stddev = []
    z = 0 

    ave = python_sma(values, n) #Calls the simple moving average function into the standard deviation function
    
    
    for a in range (1, len(values)):
        z = a+1
        dev = [(y - ave[a-n])**2 for y in values[a-n:z-1]]
        var = sum(dev)/n
        stddev.append(math.sqrt(var))

        


    return stddev

if __name__ == "__main__":    
   
    input = [0,1,2,3,4,5,6,7,8,9,10]
    
    result = python_sma(input, n = 2)
    print (result)

    results = python_stddev(input, n = 2)
    print(results)

