"""
Test a learner.  (c) 2015 Tucker Balch

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
import math
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    skip = inf.readline() # skip the first line
    data = np.array([list(map(float,s.strip()[10:].split(','))) for s in inf.readlines()]) # start to strip from the 10th string, skip all others, orignal has error

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    def exp1(leaf_max = 20):

        leaf_num = leaf_max
        rmse_train = []
        rmse_test = []
        rmse_train_bag = []
        rmse_test_bag = []

        for i in range(leaf_num):
            learner = dt.DTLearner(verbose = True,leaf_size = i ) # create a LinRegLearner
            learner.addEvidence(trainX, trainY) # train it
            # evaluate in sample
            predY = learner.query(trainX) # get the predictions
            rmse_train.append(math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))
            # evaluate out of sample
            predY = learner.query(testX) # get the predictions
            rmse_test.append(math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))

            learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size": 1}, bags = 20, boost = False, verbose = False)
            learner.addEvidence(trainX, trainY) # train it
            # evaluate in sample
            predY = learner.query(trainX) # get the predictions
            rmse_train_bag.append(math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))
            # evaluate out of sample
            predY = learner.query(testX) # get the predictions
            rmse_test_bag.append(math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))

        #----------- Fig1 for DT Overfitting
        fig1 = plt.figure(figsize=(8,4))
        plt.title('Fig1: RMSE for DT Learner')
        plt.xlabel('Leaf Size')
        plt.ylabel('RMSE')
        color = ["#9E0142","#D53E4F","#F46D43","#FDAE61","#FEE08B","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
        line1,=plt.plot(rmse_train, label = "Train",color =color[1])
        line2,=plt.plot(rmse_test, label = "Test",color =color[3])
        plt.legend()
        plt.savefig('Fig1.png')
        plt.close(fig1)

    #----------- Fig2 for  Overfitting
        fig2 = plt.figure(figsize=(8,4))
        plt.title('Fig2: RMSE for Bagging Learner')
        plt.xlabel('Leaf Size')
        plt.ylabel('RMSE')
        color = ["#9E0142","#D53E4F","#F46D43","#FDAE61","#FEE08B","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
        line3,=plt.plot(rmse_train_bag, label = "Train",color =color[5])
        line4,=plt.plot(rmse_test_bag, label = "Test",color =color[7])
        plt.legend()
        plt.savefig('Fig2.png')
        plt.close(fig2)

    #----------- Fig3 for  Overfitting
        fig3 = plt.figure(figsize=(8,4))
        plt.title('Fig3: RMSE for both DT and Bag')
        plt.xlabel('Leaf Size')
        plt.ylabel('RMSE')
        color = ["#9E0142","#D53E4F","#F46D43","#FDAE61","#FEE08B","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
        line1,=plt.plot(rmse_train, label = "DT Train",color =color[1])
        line2,=plt.plot(rmse_test, label = "DT Test",color =color[3])
        line3,=plt.plot(rmse_train_bag, label = "Bag Train",color =color[5])
        line4,=plt.plot(rmse_test_bag, label = "Bag Test",color =color[7])
        plt.legend()
        plt.savefig('Fig3.png')
        plt.close(fig3)

    def exp2(leaf_max = 20):

        leaf_num = leaf_max
        rmse_train = [] # by default for DT
        rmse_test = []
        cor_train = []
        cor_test =[]
        rmse_train_RT = []
        rmse_test_RT = []
        cor_train_RT = []
        cor_test_RT = []
        mae_train_DT = []
        mae_test_DT = []
        mae_train_RT = []
        mae_test_RT = []


        time_DT = []
        time_RT = []

        for i in range(leaf_num):
            start_time = time.time()
            learner = dt.DTLearner(verbose = True,leaf_size = i ) # create a LinRegLearner
            learner.addEvidence(trainX, trainY) # train it
            time_DT.append(time.time() - start_time)
            # evaluate in sample
            predY = learner.query(trainX) # get the predictions
            mae_train_DT.append(np.mean(np.abs(trainY-predY)))
            rmse_train.append(math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))
            # evaluate out of sample
            predY = learner.query(testX) # get the predictions
            mae_test_DT.append(np.mean(np.abs(testY-predY)))
            rmse_test.append(math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))

            start_time = time.time()
            learner = rt.RTLearner(verbose = True,leaf_size = i ) # create a LinRegLearner
            print(type(trainX))
            print(type(trainY))
            learner.addEvidence(trainX, trainY) # train it
            time_RT.append(time.time() - start_time)
            # evaluate in sample
            predY = learner.query(trainX) # get the predictions
            mae_train_RT.append(np.mean(np.abs(trainY-predY)))
            rmse_train_RT.append(math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))
            # evaluate out of sample
            predY = learner.query(testX) # get the predictions
            mae_test_RT.append(np.mean(np.abs(testY-predY)))
            rmse_test_RT.append(math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))


        #----------- Fig4 for DT Overfitting
        fig4 = plt.figure(figsize=(8,4))
        plt.title('Fig4: MAE comparison between DT and RT')
        plt.xlabel('Leaf Size')
        plt.ylabel('MAE')
        color = ["#9E0142","#D53E4F","#F46D43","#FDAE61","#FEE08B","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
        line1,=plt.plot(mae_train_DT, label = "DT Train",color =color[1])
        line2,=plt.plot(mae_test_DT, label = "DT Test",color =color[3])
        line3,=plt.plot(mae_train_RT, label = "RT Train",color =color[5])
        line4,=plt.plot(mae_test_RT, label = "RT Test",color =color[7])
        plt.legend()
        plt.savefig('Fig4.png')
        plt.close(fig4)

#        #----------- Fig5 for DT Overfitting
#        fig5 = plt.figure(figsize=(8,4))
#        plt.title('Fig5: Correlation comparison between DT and RT')
#        plt.xlabel('Leaf Size')
#        plt.ylabel('Correlation')
#        line1,=plt.plot(cor_train, label = "DT Train",color =color[5])
#        line2,=plt.plot(cor_test, label = "DT Test",color =color[7])
#        line3,=plt.plot(cor_train_RT, label = "RT Train",color =color[6])
#        line4,=plt.plot(cor_test_RT, label = "RT Test",color =color[8])
#        plt.legend()
#        plt.savefig('Fig5.png')
#        plt.close(fig5)

    #----------- Fig5 for  Overfitting
        fig5 = plt.figure(figsize=(8,4))
        plt.title('Fig5: Time distribution for DT and RT')
        plt.xlabel('Leaf Size')
        plt.ylabel('Time')
        line1,=plt.plot(time_DT, label = "DT",color =color[2])
        line2,=plt.plot(time_RT, label = "RT",color =color[4])
        plt.legend()
        plt.savefig('Fig5.png')
        plt.close(fig5)

    exp1()
    exp2()

    #---------  Below are test for myself, close 
    if(0): # test for each learner myselft, 0 means don't test'
        print(data[0,])
        print(f"{testX.shape}")
        print(f"{testY.shape}")

        # create a learner and train it
        print("#--------------------------")
        print("Start of LinRegLearner:")
        learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
        learner.addEvidence(trainX, trainY) # train it
        print(learner.author())

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print()
        print("In sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=trainY)
        print(f"corr: {c[0,1]}")

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print()
        print("Out of sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=testY)
        print(f"corr: {c[0,1]}")

        # create a learner and train it
        print()
        print("#--------------------------")
        print("Start of DTLearner:")
        #print(f"{testX.shape}")
        #print(f"{testY.shape}")
        #print(f"{trainX.shape}")
        #print(f"{trainY.shape}")
        learner.addEvidence(trainX, trainY) # train it
        print(learner.author())

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        predY = predY.astype(float)
        #print("predition.shape",predY.shape)
        #print("predition",predY.astype(float))
        #print("train.predition",trainY)
        #print("train-predition",trainY-predY.astype(float))
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print()
        print("In sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=trainY)
        print(f"corr: {c[0,1]}")

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        predY = predY.astype(float)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print()
        print("Out of sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=testY)
        print(f"corr: {c[0,1]}")
            
        # create a learner and train it
        print()
        print("#--------------------------")
        print("Start of RTLearner:")
        learner = rt.RTLearner(verbose = True) # create a LinRegLearner
        learner.addEvidence(trainX, trainY) # train it
        print(learner.author())

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        predY = predY.astype(float)
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print()
        print("In sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=trainY)
        print(f"corr: {c[0,1]}")

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        predY = predY.astype(float)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print()
        print("Out of sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=testY)
        print(f"corr: {c[0,1]}")

       # create a learner and train it
        print()
        print("#--------------------------")
        print("Start of BagLearner:")
        learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size": 1}, bags = 20, boost = False, verbose = False)
        learner.addEvidence(trainX, trainY) # train it
        print(learner.author())

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        predY = predY.astype(float)
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print()
        print("In sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=trainY)
        print(f"corr: {c[0,1]}")

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        predY = predY.astype(float)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print()
        print("Out of sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=testY)
        print(f"corr: {c[0,1]}")

       # create a learner and train it
        print()
        print("#--------------------------")
        print("Start of InsaneLearner:")
        learner = il.InsaneLearner(learner = lrl.LinRegLearner, kwargs = {}, num = 20, verbose = False)
        learner.addEvidence(trainX, trainY) # train it
        print(learner.author())

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        predY = predY.astype(float)
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print()
        print("In sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=trainY)
        print(f"corr: {c[0,1]}")

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        predY = predY.astype(float)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print()
        print("Out of sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=testY)
        print(f"corr: {c[0,1]}")
          
