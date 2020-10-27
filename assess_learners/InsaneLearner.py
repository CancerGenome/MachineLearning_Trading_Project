"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch

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
"""
import numpy as np
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl

class InsaneLearner(object):
    def __init__(self, verbose = False, num = 20, learner = lrl.LinRegLearner, kwargs = {}):
        self.verbose = verbose
        self.num = num # itimes to call
        self.learner = []
        for i in range(self.num): # repeat 20 times 
            self.learner.append(bl.BagLearner(learner = learner, kwargs = kwargs, bags =20, boost = False, verbose = False))
        self.kwargs = kwargs
        #pass # move along, these aren't the drones you're looking for

    def author(self):
        return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu

    def addEvidence(self,Xtrain,Ytrain):
        """
        @summary: Add training data to learner
        @param Xtrain: X values of data to add
        @param Ytrain: the Y training values
        Only query bagger 20 times, each time bag will query lrl for 20 times
        """
        for learner in self.learner:
            learner.addEvidence(Xtrain, Ytrain)

    def query(self,Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # same with baglearner
        Y = np.array([learner.query(Xtest) for learner in self.learner])
        #print("Y:", Y)
        #print("Y Shape:", Y.shape)
        return np.mean(Y.astype(float), axis = 0)

if __name__=="__main__":
    print("the secret clue is 'zzyzx'")

