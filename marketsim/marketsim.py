"""MC2-P1: Market simulator.

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

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu

def gtid():
    return 903459631 # replace with your GT ID number

def compute_portvals(orders_file = "./orders/orders-01.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # for test purpose
    #orders_file = "./orders/orders-01.csv"
    #start_val = 1000000
    #commission=9.95
    #impact=0.005

    # Read in file and sort
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    df = orders_df.sort_index(axis = 0)
    #orders_df = orders_df.sort_values(by = 'Symbol')

    # create data frame for holding stock each day and previous value 
    stock_name = np.unique(df['Symbol'])
    column = np.append(stock_name,['cash'])
    #date_range = pd.date_range(df.index[0], df.index[df.shape[0]-1]) # full range
    price_spy = get_data(['SPY'], pd.date_range(df.index[0], df.index[df.shape[0]-1])) # get price of SPY of all days
    #price_spy = get_data(['SPY'], np.unique(df.index)) # get price of SPY of only buy/sold day
    hold = pd.DataFrame(index = price_spy.index , columns = column) # only SPY transcation day, and a matrix for each day and each stock
    previous = pd.DataFrame(0, index=np.arange(1), columns = column) # previousd stock number and cash, one vector
    pval = pd.DataFrame(index = hold.index , columns = ['pval']) # record porfilio value
    previous.loc[0,'cash'] = start_val

    # Start to process the order and update the holding stock data frame
    for i in range(0,(df.shape[0]) ):
        symbol = df['Symbol'][i]
        share = df['Shares'][i].astype(int)
        order = df['Order'][i]
        date = df.index[i]
        price = get_data([symbol], pd.date_range(date,date)) # get price
        price_num = price[[symbol]].ix[0,0]
        if(np.isnan(hold.loc[date,symbol]) == 1):
            hold.loc[date,symbol]  = 0
        if(np.isnan(hold.loc[date,'cash']) == 1):
            hold.loc[date,symbol]  = 0
        if(order =="BUY"):
            price_num = price_num * (1+impact) # price will go up for buying
            hold.loc[date,symbol] = previous.loc[0,symbol]+ share # update each day number stock
            hold.loc[date,'cash'] = previous.loc[0,'cash'] - (price_num * share) - commission # update total cash
        if(order =="SELL"):
            price_num = price_num * (1-impact) # price will go down for buying
            hold.loc[date,symbol] = previous.loc[0,symbol] - share # update each day number stock
            hold.loc[date,'cash'] = previous.loc[0,'cash'] + (price_num * share) - commission # update total cash
        previous.loc[0,symbol] = hold.loc[date,symbol]
        previous.loc[0,'cash'] = hold.loc[date,'cash'] # end for

    # update the pvalue, by adding all stock value and cash
    hold = hold.fillna(method='ffill')
    hold = hold.fillna(value = 0)
    pval['pval'] = 0
    for stock in stock_name:
        price = get_data([stock], price_spy.index) # get price
        value = price[[stock]] * hold[[stock]]
        pval['pval'] = pval['pval'] + value[stock] 
        #print(hold)
    # Finally add Cash
    pval['pval'] = pval['pval'] + hold['cash']
    rv = pd.DataFrame(index=pval.index, data=pval.values)
    return rv
    #print(pval)

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    #start_date = dt.datetime(2008,1,1)
    #end_date = dt.datetime(2008,6,1)
    #portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
    #portvals = portvals[['IBM']]  # remove SPY
    #rv = pd.DataFrame(index=portvals.index, data=portvals.values)

    #return rv
    #return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")

if __name__ == "__main__":
    test_code()
