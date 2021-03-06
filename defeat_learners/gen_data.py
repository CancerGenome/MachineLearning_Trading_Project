"""
template for generating data to fool learners (c) 2016 Tucker Balch
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

Student Name: Yu Wang (replace with your name)
GT User ID:  ywang3564 (replace with your User ID)
GT ID: 903459631 (replace with your GT ID)

"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    X = np.zeros((100,2))
    np.random.seed(seed)
    Y = np.random.random(size = (100,))*200-100
    # Here's is an example of creating a Y from randomly generated
    # X with multiple columns
    # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3
    np.random.seed(seed)
    X[:,0] = Y - 1 + np.random.random(size=(100,)) # because +/- 1 will always around the linear line, but DT is not
    np.random.seed(seed)
    X[:,1] = Y + 1 + np.random.random(size=(100,))
    return X, Y

def best4DT(seed=1489683273):
    X = np.zeros((100,2))
    np.random.seed(seed)
    Y = np.random.random(size = (100,))*200-100
    X[:,0] = Y 
    X[:,1] = Y
    Y = Y >=0
    np.random.seed(seed)
    Y = Y + np.random.random(size=(100,))/1000
    return X, Y

def author():
    return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu
    #return 'tb34' #Change this to your user ID

if __name__=="__main__":
    print("they call me Tim.")
