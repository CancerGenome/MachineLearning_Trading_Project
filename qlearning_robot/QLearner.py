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
        if self.dyna != 0: #Orignal code
            self.t_init = np.ndarray(shape=(num_states, num_actions, num_states))
            self.t_init.fill(0.00001)
            self.t = self.t_init / self.t_init.sum(axis=2, keepdims=True)
            self.r = np.ndarray(shape=(num_states, num_actions))
            self.r.fill(-1.0)
        #if(self.dyna != 0): # Mine Code
        #    self.t = np.random.uniform(low = 0.00001 , high = 0.00001, size = (num_states, num_actions, num_states)) # T = [s,a,s']
        #    self.t_init = self.t
        #    self.t = self.t/sum(self.t_init[0,0,:]) # this only works for initialization
        #    self.r = np.random.uniform(low = 0, high = 0, size = (num_states, num_actions)) # R = [s,a]
            # this is the probability of T = [s,a,s'] where s and a are fixed
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
        action = np.argmax(self.q[s_prime,])
        if rand.random() <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        #else:
        #action = self.querysetstate(s_prime)

        self.rar = self.rar * self.radr

        if(self.dyna != 0): # original copy code
            self.t_init[self.s, self.a, s_prime] = self.t_init[self.s, self.a, s_prime] + 1
            self.t = self.t_init / self.t_init.sum(axis=2, keepdims=True)
            self.r[self.s, self.a] = (1 - self.alpha) * self.r[self.s, self.a] + (self.alpha * r)
            # Mine Code
            #self.t_init[self.s, self.a, s_prime] = self.t_init[self.s, self.a, s_prime] + 1 # increment here
            #self.t[self.s, self.a,:] = self.t_init[self.s, self.a,:]/sum(self.t_init[self.s, self.a,:]) # update t with new value, make sure each row add up to one
            ##self.t = self.t / self.t.sum(axis=2, keepdims=True)
            #self.r[self.s,self.a] = (1-self.alpha) * self.r[self.s, self.a] + self.alpha * r  # update reward  
            # iteration and update the q table
            for i in range(0,self.dyna): # original copy code
                a_dyna = np.random.randint(low=0, high=self.num_actions)
                s_dyna = np.random.randint(low=0, high=self.num_states)
                # infer s' from T
                s_prime_dyna = np.random.multinomial(1, self.t[s_dyna, a_dyna,]).argmax()
                # compute R from s and a
                r = self.r[s_dyna, a_dyna]
                # update Q
                self.q[s_dyna, a_dyna] = (1 - self.alpha) * self.q[s_dyna, a_dyna] + self.alpha * (r + self.gamma * np.max(self.q[s_prime_dyna,]))

                # mine code
                #s_tmp = rand.randint(0, self.num_states - 1 )
                #a_tmp = rand.randint(0, self.num_actions - 1)
                #s_prime_tmp = self.t[s_tmp, a_tmp,].argmax()
                #s_prime_tmp = np.random.multinomial(1, self.t[s_tmp, a_tmp,]).argmax()
                #r = self.r[s_tmp, a_tmp]
                #self.q[s_tmp, a_tmp] =            (1-self.alpha) * self.q[s_tmp, a_tmp] + self.alpha * (r + self.gamma * np.max(self.q[s_prime_tmp,:]))

        self.s = s_prime
        self.a = action
        if self.verbose: 
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action

    def author(self):
        return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu

if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
