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

class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
    #    pass # move along, these aren't the drones you're looking for

    def author(self):
        return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu

    def addEvidence(self,Xtrain,Ytrain):
        """
        @summary: Add training data to learner
        @param Xtrain: X values of data to add
        @param Ytrain: the Y training values
        """
        # build and save the model
        #print("X:",Xtrain)
        #print("Y:",Ytrain)
        #should extend Ytrain to 2D with  b[:,None] , https://stackoverflow.com/questions/41989950/numpy-array-concatenate-valueerror-all-the-input-arrays-must-have-same-number?rq=1
        self.model = self.build_tree( np.concatenate((Xtrain, Ytrain[:,None]), axis = 1) ) 
        if self.verbose == 'True':
            print("Model Done")

    def build_tree(self, data):
        # get the sample and feature count
        x = data[:,0:-1]
        y = data[:,-1]
        sample_n = x.shape[0]
        feature_n = x.shape[1]
        if(sample_n <= self.leaf_size) or (np.all(y==y[0])) : #means all Y are same or leaf size smaller than others
        #if(sample_n <= self.leaf_size or np.unique(y).shape[0] == 1): #means all Y are same or leaf size smaller than others
            return np.array([["Leaf", np.mean(y), "NA","NA"]],dtype = object) # the branch end
        else:
            # Random assign features,
            random_index = np.random.randint(0, feature_n)
            #print(f"random_index:{random_index}")
            # Here choose the random value, rather than the most correlated value. 
            random_sample = [np.random.randint(0,sample_n), np.random.randint(0, sample_n)]
            while random_sample[0] == random_sample[1]: # get two random number until they are different
                random_sample = [np.random.randint(0,sample_n), np.random.randint(0, sample_n)]

            split_value = np.mean([x[random_sample[0], random_index],x[random_sample[1], random_index]])

            if(np.all(data[:,random_index] <= split_value)):
                return np.array([["Leaf", np.mean(y), "NA","NA"]], dtype = object) # the branch end
            left_tree = self.build_tree(data[data[:,random_index] <= split_value])
            right_tree = self.build_tree(data[data[:,random_index] > split_value])
            root = np.array([[random_index,split_value,1, left_tree.shape[0] + 1]], dtype = object)
            return(np.concatenate((root, left_tree, right_tree), axis = 0))

    def query(self,Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        Y = []
        for point in Xtest:
            #print("Return:", self.tree_search(point,0))
            Y.append(self.tree_search(point,0))
        #print("Shape of Point: ", np.shape(point))
        #print("Shape of Y: ", np.shape(Y))
        return np.asarray(Y)
        
    def tree_search(self,point,row):
        """
        point is the search key
        row is the starting row number
        return: Y predicted value
        """
        feature,split_value,left_point, right_point = self.model[row, 0:4]
        if feature == "Leaf":
            #print("Split value",split_value) 
            return split_value
        # if point <= split_value, go to the left
        elif point[feature] <= split_value:
            prediction = self.tree_search(point, row + int(left_point))
        else: 
            prediction = self.tree_search(point, row + int(right_point))
        return prediction

if __name__=="__main__":
    print("the secret clue is 'zzyzx'")

