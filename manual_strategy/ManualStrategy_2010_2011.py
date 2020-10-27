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
#from marketsimcode import compute_portvals

def author():
    return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu

def gtid():
    return 903459631 # replace with your GT ID number

def BB(price, window = 15):
    sma = price.rolling(window).sum()/window
    std = price.rolling(window).std()
    bb1 = sma - 2* std
    bb2 = sma + 2* std
    bb = pd.concat([bb1,bb2], axis = 1)
    bb.columns = ['LowBB',"HighBB"]
    return(bb)

def testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000, return_sv = 0, commission = 9.95, impact = 0.005):
    ### For Test 
    #symbol = 'JPM'
    #sd=dt.datetime(2008, 1, 1)
    #ed=dt.datetime(2009,12,31)
    #sv = 100000
    symbol_list = list(symbol.split(" "))
    price = get_data(symbol_list, pd.date_range(sd,ed) ) 
    price = price[symbol_list]
    ### Create Data Frame for Record
    Hold = pd.DataFrame(index=price.index)
    Hold['Stock'] = 0
    Hold['Buy'] = 0 
    Hold['Cash'] = sv 
    Hold['SV'] = sv 
    n = price.shape[0] - 1
    ### Get Indicator of BB
    bb = BB(price, window = 15)
    ### Process each day
    for i in range(15,n+1):
        if i == n:
            Hold['Stock'][i] = 0 
        elif (i< n and  price[symbol][i] >= bb['HighBB'][i]):
            Hold['Stock'][i] = -1000
        elif (i < n and price[symbol][i] <= bb['LowBB'][i]):
            Hold['Stock'][i] = +1000
        Hold['Buy'][i] = Hold['Stock'][i] - Hold['Stock'][i-1]

        if Hold['Buy'][i] > 0:
            Hold['Cash'][i] = Hold['Cash'][i-1] - Hold['Buy'][i] * price[symbol][i] * (1+impact) - commission
        elif Hold['Buy'][i] < 0: # mean this is SALE
            Hold['Cash'][i] = Hold['Cash'][i-1] - Hold['Buy'][i] * price[symbol][i] * (1-impact) - commission
        elif Hold['Buy'][i] == 0:
            Hold['Cash'][i] = Hold['Cash'][i-1] 
        Hold['SV'][i] = Hold['Cash'][i] + Hold['Stock'][i] * price[symbol][i]
    if return_sv == 1:
        return Hold['SV']
    else: 
        return Hold['Buy']

def benchmark(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):
    price_bench = get_data([symbol], pd.date_range(sd, ed) ) 
    price_bench= price_bench[[symbol]]
    hold_cash = sv - price_bench.iloc[0] * 1000
    bench_value = hold_cash + price_bench * 1000
    return(bench_value)

def test_code():
    ### Prepare input
    symbol = 'JPM'
    sd=dt.datetime(2010, 1, 1)
    ed=dt.datetime(2011,12,31)
    sv = 100000
    symbol_list = list(symbol.split(" "))
    price = get_data(symbol_list, pd.date_range(sd,ed) ) 
    price = price[symbol_list]
    ### Calculate personal stratgy and benchmark, then normalized
    personal = testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv, return_sv = 1)
    personal_buy = testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv, return_sv = 0)
    bench = benchmark(symbol = symbol, sd = sd, ed = ed, sv = sv)
    daily_return_personal = personal/personal.shift(1)
    daily_return_bench = bench/bench.shift(1)
    personal = personal/sv
    bench = bench/sv

    #### Plot Main Figures, including benchmark, BB,  mannual  
    fig = plt.figure(figsize=(8,4))
    plt.title('Mannual Rule (BB) Strategy')
    plt.xlabel('Date')
    plt.ylabel('Normalized JPM Price')
    #plt.xlim(0,300)
    plt.ylim(0.7,1.4)
    color = ["#9E0142","#D53E4F","#F46D43","#FDAE61","#FEE08B","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
    bb = BB(price,window = 15)
    bb = bb/price[symbol][0]
    line1,=plt.plot(personal,label = 'Mannual Rule', color = 'red')
    line2,=plt.plot(bench, label = "Benchmark", color ='green')
    #line3,=plt.plot(bb['LowBB'], linestyle = "--", label = "BB Low Range", color = color[8])
    #line4,=plt.plot(bb['HighBB'], linestyle = "--", label = 'BB High Range', color = color[9])

    # Draw vertical lines 
    n = bench.shape[0] 
    for i in range(0,n):
        if personal_buy[i] == 1000:
            plt.axvline(x=personal_buy.index[i], color ='blue', linestyle ="--", linewidth = 0.5)
        if personal_buy[i] == -1000:
            plt.axvline(x=personal_buy.index[i], color ='red', linestyle ="--",linewidth = 0.5)
    #plt.legend(handles=[line1,line2,line3,line4], loc=2)
    plt.legend(handles=[line1,line2], loc=2)
    plt.savefig('Mannual_Rule_2010_2011.png')
    plt.close(fig)

    #### Print CR, STD, Mean

    n = bench.shape[0] - 1
    print('Benchmark Cumulative Return:', bench[symbol][n]-1  )
    print('Benchmark STD:', np.std(daily_return_bench.iloc[1:])  )
    print('Benchmark Mean:', np.mean(daily_return_bench.iloc[1:])-1  )
    print('personal Cumulative Return:', personal[n]-1  )
    print('personal STD:', np.std(daily_return_personal.iloc[1:])  )
    print('personal Mean:', np.mean(daily_return_personal.iloc[1:]) -1 )

if __name__ == "__main__":
    test_code()
