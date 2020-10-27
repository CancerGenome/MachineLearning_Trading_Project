"""
Borrow from my Own Script and update
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
import StrategyLearner as st
from marketsim import compute_portvals
from ManualStrategy import testPolicy,benchmark
from indicators import BB

def author():
    return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu

def gtid():
    return 903459631 # replace with your GT ID number

def test_code(commission = 0, impact = 0, filename = 'Mannual_Rule_and_Strategy_Learner.png'):

    ### Prepare input
    symbol = 'JPM'
    sd=dt.datetime(2008, 1, 1)
    ed=dt.datetime(2009,12,31)
    sv = 100000
    symbol_list = list(symbol.split(" "))
    price = get_data(symbol_list, pd.date_range(sd,ed) ) 
    price = price[symbol_list]

    ### Calculate Manual  stratgy and benchmark, then normalized
    personal = testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv, return_sv = 1,commission = commission, impact = impact)
    personal_buy = testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv, return_sv = 0,commission = commission, impact = impact)
    bench = benchmark(symbol = symbol, sd = sd, ed = ed, sv = sv)
    daily_return_personal = personal/personal.shift(1)
    daily_return_bench = bench/bench.shift(1)
    personal = personal/sv
    bench = bench/sv

    ### Learner from Machine Learning, use impact as zero
    stlearner = st.StrategyLearner(verbose = False, impact=0.0) # initialization
    stlearner.addEvidence(symbol="JPM",sd=sd,ed=ed,sv=sv) # train data
    trade = stlearner.testPolicy(symbol="JPM",sd=sd,ed=ed,sv=sv) # get output
    # calculate the markert value of st
    cash = trade.copy()
    cash.values[:]  = 0
    total = trade.copy()
    total.values[:]  = 0
    pre_cash = sv
    stock_hold = 0 
    for i in range(0, price.shape[0]):
        if trade.values[i] > 0: # buy price will go up
            price_tmp = price.values[i] * ( 1 + impact)
            cash.values[i] = pre_cash - price_tmp *  trade.values[i] - commission
        elif trade.values[i] < 0: #sale price will go down
            price_tmp = price.values[i] * ( 1 - impact)
            cash.values[i] = pre_cash - price_tmp *  trade.values[i] - commission
        elif trade.values[i] == 0: #sale price will go down
            cash.values[i] = pre_cash - commission
        pre_cash = cash.values[i]
        stock_hold = stock_hold + trade.values[i]
        total.values[i] = cash.values[i] + price.values[i] * stock_hold
        #print(i,total.values[i], stock_hold, price.values[i], cash.values[i])
    total = total/sv
    daily_return_total = total/total.shift(1)

    #### Plot Main Figures, including benchmark, BB,  mannual  
    fig = plt.figure(figsize=(8,4))
    plt.title('Strategy comparison between mannual, strategy learner, and benchmark')
    plt.xlabel('Date')
    plt.ylabel('Normalized JPM Price')
    #plt.xlim(0,300)
    plt.ylim(-2,4)
    color = ["#9E0142","#D53E4F","#F46D43","#FDAE61","#FEE08B","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
    bb = BB(price,window = 15)
    bb = bb/price[symbol][0]
    line1,=plt.plot(personal,label = 'Mannual Rule', color = 'red')
    line2,=plt.plot(bench, label = "Benchmark", color ='green')
    line3,=plt.plot(total, label = "Strategy Learner", color =color[8])
    #line3,=plt.plot(bb['LowBB'], linestyle = "--", label = "BB Low Range", color = color[8])
    #line4,=plt.plot(bb['HighBB'], linestyle = "--", label = 'BB High Range', color = color[9])

    # Draw vertical lines 
    #n = bench.shape[0] 
    #for i in range(0,n):
    #    if personal_buy[i] == 1000:
    #        plt.axvline(x=personal_buy.index[i], color ='blue', linestyle ="--", linewidth = 0.5)
    #    if personal_buy[i] == -1000:
    #        plt.axvline(x=personal_buy.index[i], color ='red', linestyle ="--",linewidth = 0.5)
    #plt.legend(handles=[line1,line2,line3,line4], loc=2)
    plt.legend(handles=[line1,line2,line3], loc=2)
    plt.savefig(filename)
    plt.close(fig)

    #### Print CR, STD, Mean

    n = bench.shape[0] - 1
    print('Benchmark Cumulative Return:', bench[symbol][n]-1  )
    print('Benchmark STD:', np.std(daily_return_bench.iloc[1:])  )
    print('Benchmark Mean:', np.mean(daily_return_bench.iloc[1:])-1  )
    print('personal Cumulative Return:', personal[n]-1  )
    print('personal STD:', np.std(daily_return_personal.iloc[1:])  )
    print('personal Mean:', np.mean(daily_return_personal.iloc[1:]) -1 )
    print('Strategy Cumulative Return:', total[symbol][n]-1  )
    print('Strategy STD:', np.std(daily_return_total.iloc[1:])  )
    print('Strategy Mean:', np.mean(daily_return_total.iloc[1:])-1  )

if __name__ == "__main__":
    test_code()
