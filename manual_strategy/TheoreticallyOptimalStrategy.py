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

def testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000, return_sv = 0):
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
    ### Process each day
    for i in range(0,n+1):
        if i == n:
            Hold['Stock'][i] = 0 
        elif (i< n and  price[symbol][i] > price[symbol][i+1]):
            Hold['Stock'][i] = -1000
        elif (i < n and price[symbol][i] <= price[symbol][i+1]):
            Hold['Stock'][i] = +1000
        Hold['Buy'][i] = Hold['Stock'][i] - Hold['Stock'][i-1]
        Hold['Cash'][i] = Hold['Cash'][i-1] - Hold['Buy'][i] * price[symbol][i]
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
    symbol = 'JPM'
    sd=dt.datetime(2008, 1, 1)
    ed=dt.datetime(2009,12,31)
    sv = 100000
    best = testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv, return_sv = 1)
    bench = benchmark(symbol = symbol, sd = sd, ed = ed, sv = sv)
    daily_return_best = best/best.shift(1)
    daily_return_bench = bench/bench.shift(1)
    best = best/sv
    bench = bench/sv
    #### Plot Figures
    fig = plt.figure(figsize=(8,4))
    plt.title('Theoretical Best Strategy')
    plt.xlabel('Date')
    plt.ylabel('Normalized JPM Price')
    #plt.xlim(0,300)
    plt.ylim(0,8)
    line1,=plt.plot(best,label = 'Theoretical Best', color = 'red')
    line2,=plt.plot(bench, label = "Benchmark", color ='green')
    plt.legend(handles=[line1,line2], loc=2)
    plt.savefig('TheoreticalBest.png')
    plt.close(fig)

    #### Print CR, STD, Mean
    n = bench.shape[0] - 1

    print('Benchmark Cumulative Return:', bench[symbol][n]-1  )
    print('Benchmark STD:', np.std(daily_return_bench.iloc[1:])  )
    print('Benchmark Mean:', np.mean(daily_return_bench.iloc[1:])-1  )
    print('Best Cumulative Return:', best[n]-1  )
    print('Best STD:', np.std(daily_return_best.iloc[1:])  )
    print('Best Mean:', np.mean(daily_return_best.iloc[1:]) -1 )

if __name__ == "__main__":
    test_code()
