"""
Project 6: Manual Strategy
Student Name: Yu Wang 
GT User ID:  ywang3564 
GT ID: 903459631
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt

def author():
    return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu

def gtid():
    return 903459631 # replace with your GT ID number

def mem(price, n = 1): # do not use this
    mem1 = price.values[0:(nrow-1)]
    mem = ( price[1:nrow]/mem1 ) - 1 
    return(mem)

def SMA(price, window = 5):
    sma = price.rolling(window).sum()/window
    return(sma)

def BB(price, window = 5):
    sma = price.rolling(window).sum()/window
    std = price.rolling(window).std()
    bb1 = sma - 2* std
    bb2 = sma + 2* std
    bb = pd.concat([bb1,bb2], axis = 1)
    bb.columns = ['LowBB',"HighBB"]
    return(bb)

def PP(price, window = 5):
    maxV = price.rolling(window).max()
    minV = price.rolling(window).min()
    return((maxV + minV + price)/3)

def indicator_plot():

    price_NVTA = get_data(['NVTA'], pd.date_range('2019-01-01','2019-11-01'), addSPY=False ) # get price of SPY of all days
    price_NVTA = price_NVTA.dropna()
    price_NVTA = price_NVTA[['NVTA']]
    price_NVTA = price_NVTA/price_NVTA.iloc[0]
    sma = SMA(price_NVTA, window = 15)
    bb = BB(price_NVTA, window = 15)
    pp = PP(price_NVTA, window = 15)

#---- Plot SMA
    fig = plt.figure(figsize=(8,4))
    plt.title('Smooth Mean Value(SMA)')
    plt.xlabel('Date')
    plt.ylabel('Normalized NVTA Price')
    #plt.xlim(0,300)
    plt.ylim(0,3)
    color = ["#9E0142","#D53E4F","#F46D43","#FDAE61","#FEE08B","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
    line1,=plt.plot(price_NVTA,label = 'Price', color = color[1])
    line2,=plt.plot(sma, linestyle = "--", label = "SMA", color = color[8])
    line3,=plt.plot(sma/price_NVTA, linestyle = "--", label = 'SMA/Price', color = color[7])
    plt.legend(handles=[line1,line2,line3], loc=4)    
    plt.savefig('SMA.png')
    plt.close(fig)

#---- Plot BB
    fig = plt.figure(figsize=(8,4))
    plt.title('Bollinger Band (BB)')
    plt.xlabel('Date')
    plt.ylabel('Normalized NVTA Price')
    #plt.xlim(0,300)
    #plt.ylim(-256,100)
    plt.ylim(0,3)
    color = ["#9E0142","#D53E4F","#F46D43","#FDAE61","#FEE08B","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
    line1,=plt.plot(price_NVTA,label = 'Price', color = color[1])
    line2,=plt.plot(bb['LowBB'], linestyle = "--", label = "BB Low Range", color = color[8])
    line3,=plt.plot(bb['HighBB'], linestyle = "--", label = 'BB High Range', color = color[9])
    plt.legend(handles=[line1,line2,line3], loc=4)    
    plt.savefig('BB.png')
    plt.close(fig)

#---- Plot Pivot Point
    fig = plt.figure(figsize=(8,4))
    plt.title('Pivot Point(PP)')
    plt.xlabel('Date')
    plt.ylabel('Normalized NVTA Price')
    #plt.xlim(0,300)
    #plt.ylim(-256,100)
    plt.ylim(0,3)
    color = ["#9E0142","#D53E4F","#F46D43","#FDAE61","#FEE08B","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
    line1,=plt.plot(price_NVTA,label = 'Price', color = color[1])
    line2,=plt.plot(pp, linestyle = "--", label = "Pivot Point", color = color[8])
    line3,=plt.plot(pp/price_NVTA, linestyle = "--", label = 'Pivot Point/Price', color = color[7])
    plt.legend(handles=[line1,line2,line3], loc=4)    
    plt.savefig('PP.png')
    plt.close(fig)


if __name__ == "__main__":
    indicator_plot()
