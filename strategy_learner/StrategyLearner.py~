"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Yu Wang 
GT User ID:  ywang3564 
GT ID: 903459631
"""

import datetime as dt
import pandas as pd
import util as ut
import random
import RTLearner as rt
import BagLearner as bl
from indicators import SMA,BB,PP
import numpy as np

class StrategyLearner(object):
    def author(self):
        return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu
    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        # Bag Learner with random forest, this is same with previous assess learner
        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":5}, bags = 20, boost = False, verbose = False)
        #######
    # this method should create a BagLearner and RTLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):
        #######
        # example usage of the old backward compatible util function
        syms=[symbol] # this will return a list, different with directly syms = symbol
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        #######
        # add your code to do learning here, get all indicators
        # PP is Pivot Point which we have used in assigment 7, manual strategy
        pp = PP(prices)
        pp = pp.rename(columns={symbol:'PP'})
        # sma is Simple moving average 
        sma = SMA(prices)
        sma = sma.rename(columns={symbol:'SMA'})
        # bb is Bollinger Bands
        bb = BB(prices) # have two columns: LowBB, HighBB
        #######
        # Start to create train X and train Y:
        indicators = pd.concat((pp,sma,bb),axis=1)
        indicators.fillna(0,inplace=True)
        indicators=indicators[:-5] # remove last five days because of last five days have no future price
        trainX = indicators.values
        #######
        # use future five days as prediction of Y 
        trainY = np.zeros(shape= (prices.shape[0]-4)) 
        for i in range(prices.shape[0]-5):
            ret = (prices.ix[i+5,syms]/prices.ix[i,syms]).values - 1.0
            #print("Ret:",ret)
            if ret > self.impact:
             #   trainY.append(1)
            #if ret > 0:
            #    trainY.ix[i,syms] = 1
                trainY[i] = 1
            elif ret < -1 * self.impact:
             #   trainY.append(-1)
            #elif ret < 0:
                #trainY.ix[i,syms] = -1
                trainY[i] = -1
            else:
                #trainY.ix[i,syms] = 0
                trainY[i] = 0
                #trainY.append(0)
            trainY = np.asarray(trainY)
            ###
        # Training for bagging learner, and random forest
        self.learner.addEvidence(trainX,trainY)
        if self.verbose: print(prices)
    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        #####
        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        syms=[symbol] # this will return a list, different with directly syms = symbol
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
     ###`
     # PP is Pivot Point which we have used in assigment 7, manual strategy
        pp = PP(prices)
        pp = pp.rename(columns={symbol:'PP'})
        # sma is Simple moving average 
        sma = SMA(prices)
        sma = sma.rename(columns={symbol:'SMA'})
        # bb is Bollinger Bands
        bb = BB(prices) # have two columns: LowBB, HighBB
        #######
        # Start to create test value 
        indicators = pd.concat((pp,sma,bb),axis=1)
        indicators.fillna(0,inplace=True)
        indicators=indicators[:-5] # remove last five days because of last five days have no future price
        tradeX = indicators.values
        tradeY = self.learner.query(tradeX)
        #print(tradeY.shape) # this is n - 5
        #######
        #trades = prices_all[[symbol,]]  # only portfolio symbols
        trades = prices  # only portfolio symbols
        trades.values[:,:] = 0 # set them all to nothing
        trades.values[0,:] = 1000 # add a BUY at the start
        #print("Trade", trades.values.shape)
        previous = 1000 # indicate previous holding
        for i in range(1,prices.shape[0]-5): # should be n - 5, because last five does not have future value
            if tradeY[i] > 0:
                trades.values[i,:] = 1000 - previous
                previous = 1000
            elif tradeY[i] < 0:
                trades.values[i,:] = -1000 - previous
                previous = -1000
            elif tradeY[i] == 0:
                trades.values[i,:] = 0 - previous
                previous = 0
        if previous == 1000:
            trades.values[-1,:] = -1000 #exit on the last day
        elif previous == -1000:
            trades.values[-1,:] = 1000 #exit on the last day
        if self.verbose: print(type(trades)) # it better be a DataFrame!
        if self.verbose: print(trades)
        if self.verbose: print(prices_all)
        return trades
    ####### Test for Main
if __name__=="__main__":
    print("One does not simply think up a strategy")
    st = StrategyLearner()
    st.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    st.testPolicy(symbol="JPM",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
