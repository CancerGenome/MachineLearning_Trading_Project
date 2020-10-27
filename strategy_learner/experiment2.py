"""
Borrow from my Own Script and update, similar with experiment1,
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
from experiment1 import test_code

def author():
    return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu

def gtid():
    return 903459631 # replace with your GT ID number

if __name__ == "__main__":
    test_code(impact = 0.005, filename = 'Mannual_Rule_and_Strategy_Learner_impact0.005.png')
    test_code(impact = 0.05, filename = 'Mannual_Rule_and_Strategy_Learner_impact0.05.png')
