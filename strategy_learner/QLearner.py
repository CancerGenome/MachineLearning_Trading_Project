"""
Template for implementing QLearner  (c) 2015 Tucker Balch

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

import numpy as np
import random as rand

class QLearner(object):
    def __init__(self, \
        num_states=100,
        num_actions = 4,
        alpha = 0.2,
        gamma = 0.9, 
        rar = 0.5, 
        radr = 0.99,
        dyna = 0,
        verbose = False):

        """
        @summary: Initialization of Q table
        @num_states: number of state 
        @num_actions: number of action 
        @alpha: learn rate, probability change to a new Q, default: 0.2
        @gamma: discount rate, default: 0.9
        @rar: random action rate, probability to choose random action default: 0.5
        @radr: random action decay rate, rar = rar * radr for each update, default: 0.99
        @dyna: number of dyna update, default: 0 
        @verbose: print debug info or not
        @param s: The new state
        """
        # For testing purpose
        #num_states=100
        #num_actions = 4
        #alpha = 0.2
        #gamma = 0.9
        #rar = 0.5
        #radr = 0.99
        #dyna = 0
        #verbose = False
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma # discount rate
        self.rar = rar
        self.radr = radr
        self.verbose = verbose
        # Define the Q table here with random values, will update it later
        self.q = np.random.uniform(low = 0, high = 0, size = (num_states, num_actions))
        # Defint the T and R table for Dyna Learning, where T is the transition and R is the reward matrix
        self.dyna = dyna
        if(self.dyna != 0): # Mine Code
            print("Faile to implement DYNA on test 8 and 9 , give up")

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = self.q[s,:].argmax()
        if rand.random() <= self.rar:
            action = rand.randint(0, self.num_actions-1)
        #else:
            #print("Action:",action)
        #self.a = action
        if self.verbose: 
            print(f"s = {s}, a = {action}")
        return(action)

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        # New Q table value equals to = (1-alpha) * old value + alpha * (reward + discount_factor * max (new state,  S_prime))
        self.q[self.s, self.a] =  (1-self.alpha) * self.q[self.s, self.a]  + self.alpha * (r + self.gamma * np.max(self.q[s_prime,:]) )
        # get new action based on current s_prime without updating Q table
        action = self.querysetstate(s_prime)
        self.rar = self.rar * self.radr

        if(self.dyna != 0): # original copy code
            print("Faile to implement DYNA on test 8 and 9 , give up")
            # FAILED to solve problem 9 and 10, give up here

        self.s = s_prime
        self.a = action
        if self.verbose: 
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action

    def author(self):
        return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu

if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
