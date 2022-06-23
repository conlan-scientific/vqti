#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:24:47 2022

 Exploring the various bars and the triple barrier method
 
 Thoughts: 
     * Excited to try the bar that accounts for volume 
     * Compare and contrast performance with linear model on time bar
     * Compare and contrast optimization results on time bar vs. other bars 
     
Do we have transaction level data? 

@author: ellenyu
"""

# Time Bars 
## this is our OCHL data 
## doesn't account for differnt volumes of activit e.g. the hour following following open is much more active
## time-sampled series violates the needs of a linear model such as non-correlated features, non-heteroscedascity, aand normality of returns 
 
# Tick Bars 
## how to: extract each time a pre-defined number of transactions takes place e.g. 1,000 ticks (think, information arrival)

#Volume Bars 
## Problem with tick bars is... one transaction can be size 1 or size 10. Instead, we should sample every time a pre-defined amount of the security's
##units (shares, futures contracts, etc.) 

#Dollar Bars
## Problem with tick and volume bars is the high oscilliation esp. for an asset with significant price fluctuations e.g. crypto 
## So, sample bars in terms of dollar value excahnged e.g. $1000 worth of stocks
## Note, instead of a fix bar size, Prado suggests a dynamnic bar size as a function of the free-floating market cap of a company or
## the outstanding amount of issed debt (as a result of new issuances or buy backs of stocks or securities)

#Information-driven Bars 
## sample when information arrives to the market
## the idea is through imbalanced signed volumes, we can detected the arrival of informed traders

#Tick Imbalance Bars 
#Volume/Dollar Imbalance Bars 

#Tick Runs Bars 
#Volume/Dollar Runs Bars 
