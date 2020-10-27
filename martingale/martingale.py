"""Assess a betting strategy.

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
import pandas as pd
from pandas import DataFrame, Series
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def author():
    return 'ywang3564' # Name Yu Wang, email yuwang11@gatech.edu

def gtid():
    return 903459631 # replace with your GT ID number

def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result

def test_code(n = 1, bank_roll = 1e+1000): # the max number to lose is 1e+999 if each time failed
    win_prob = 18/38 # set appropriately to the probability of a win
    np.random.seed(gtid()) # do this only once
    #print(get_spin_result(win_prob)) # test the roulette spin

    # add your code here to implement the experiments
    winnings = np.zeros((n, 1000))
    winnings.fill(80)
    #while episode_winnings < 80 and episode_winnings >= 0 :

    for j in range(1,n-1):
        episode_winnings = 0 
        winnings[j,0] = 0 # starting number
        i = 1

        while (episode_winnings < 80): 
            won = False
            bet_amount = 1
            if(episode_winnings < -1 * bank_roll):
                #print('Break1')
                winnings[j,i] = -1 * bank_roll
                winnings[j,i:999] = -1 * bank_roll
                break

            while not won: # stop until won
                #if(episode_winnings >= -1 * bank_roll):
                won = get_spin_result(win_prob)
                if won == True:
                    episode_winnings = episode_winnings + bet_amount
                else:
                    episode_winnings = episode_winnings - bet_amount
                    bet_amount = bet_amount * 2
                    if bet_amount*2>=episode_winnings+bank_roll:
                        bet_amount=episode_winnings+bank_roll
                    else:
                        bet_amount*=2

                if(episode_winnings >= 80):
                    episode_winnings = 80

                if(episode_winnings < -1 * bank_roll):
                    #print('Break2')
                    winnings[j,i] = -256
                    winnings[j,i:999] = -256
                    break

                winnings[j,i] = episode_winnings # record each time
                i = i + 1
                #else:
                #    print('Here')
                #    winnings[j,i] = -256
                #    winnings[j,i:999] = -256
                #    break
                #print('Won:', won, 'Episode Winning',episode_winnings ,'Bet Amount', bet_amount)
#print(winnings)

    return winnings

if __name__ == "__main__":
    
    #----- Figure 1
    n = 10
    #winnings = test_code(n = n,bank_roll= 256)
    winnings = test_code(n = n)
    fig1 = plt.figure(figsize=(8,4))
    cmap = matplotlib.cm.get_cmap('Spectral')
    plt.title('Fig1')
    plt.xlabel('Iterations')
    plt.ylabel('Winnings')
    plt.xlim(0,300)
    plt.ylim(-256,100)
    color = ["#9E0142","#D53E4F","#F46D43","#FDAE61","#FEE08B","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
    for j in range(1,n):
        plt.plot(winnings[j,], color = color[j])
    plt.savefig('fig1.png')
    plt.close(fig1)

    #------ Figure 2 
    n = 1000
    winnings = test_code(n = n)
    fig2 = plt.figure(figsize=(8,4))
    plt.title('Fig2')
    plt.xlabel('Iterations')
    plt.ylabel('Winnings')
    plt.xlim(0,300)
    plt.ylim(-256,100)
    line1,=plt.plot(np.mean(winnings, axis = 0),label='Mean')
    line2,=plt.plot(np.mean(winnings, axis = 0) + np.std(winnings, axis = 0 ), linestyle = "--", label = "Mean+Std")
    line3,=plt.plot(np.mean(winnings, axis = 0) - np.std(winnings, axis = 0 ), linestyle = "--", label = 'Mean-Std')
    plt.legend(handles=[line1,line2,line3], loc=4)    
    plt.savefig('fig2.png')
    plt.close(fig2)

    #----- Figure 3
    fig3 = plt.figure(figsize=(8,4))
    plt.title('Fig3')
    plt.xlabel('Iterations')
    plt.ylabel('Winnings')
    plt.xlim(0,300)
    plt.ylim(-256,100)
    line1,=plt.plot(np.median(winnings, axis = 0),label='Median')
    line2,=plt.plot(np.median(winnings, axis = 0) + np.std(winnings, axis = 0 ), linestyle = "--", label = "Median+Std")
    line3,=plt.plot(np.median(winnings, axis = 0) - np.std(winnings, axis = 0 ), linestyle = "--", label = 'Median-Std')
    plt.legend(handles=[line1,line2,line3], loc=4)    
    plt.savefig('fig3.png')
    plt.close(fig3)

    #---- Figure 6
    fig6 = plt.figure(figsize=(8,4))
    plt.title('Fig6')
    plt.xlabel('Iterations')
    plt.ylabel('Winnings')
    plt.xlim(0,300)
    #plt.ylim(-256,100)
    plt.plot(np.std(winnings, axis = 0 ))
    plt.savefig('fig6.png')
    plt.close(fig6)

    #------ Figure 4
    n = 1000
    winnings = test_code(n = n, bank_roll = 256)
    fig4 = plt.figure(figsize=(8,4))
    plt.title('Fig4')
    plt.xlabel('Iterations')
    plt.ylabel('Winnings')
    plt.xlim(0,300)
    plt.ylim(-256,100)
    line1,=plt.plot(np.mean(winnings, axis = 0),label='Mean')
    line2,=plt.plot(np.mean(winnings, axis = 0) + np.std(winnings, axis = 0 ), linestyle = "--", label = "Mean+Std")
    line3,=plt.plot(np.mean(winnings, axis = 0) - np.std(winnings, axis = 0 ), linestyle = "--", label = 'Mean-Std')
    plt.legend(handles=[line1,line2,line3], loc=4)    
    plt.savefig('fig4.png')
    plt.close(fig4)

    #----- Figure 5
    fig5 = plt.figure(figsize=(8,4))
    plt.title('Fig5')
    plt.xlabel('Iterations')
    plt.ylabel('Winnings')
    plt.xlim(0,300)
    plt.ylim(-256,100)
    line1,=plt.plot(np.median(winnings, axis = 0),label='Median')
    line2,=plt.plot(np.median(winnings, axis = 0) + np.std(winnings, axis = 0 ), linestyle = "--", label = "Median+Std")
    line3,=plt.plot(np.median(winnings, axis = 0) - np.std(winnings, axis = 0 ), linestyle = "--", label = 'Median-Std')
    plt.legend(handles=[line1,line2,line3], loc=4)    
    plt.savefig('fig5.png')
    plt.close(fig5)

  #---- Figure 7
    fig7 = plt.figure(figsize=(8,4))
    plt.title('Fig7')
    plt.xlabel('Iterations')
    plt.ylabel('Winnings')
    plt.xlim(0,300)
    #plt.ylim(-256,100)
    plt.plot(np.std(winnings, axis = 0 ))
    plt.savefig('fig7.png')
    plt.close(fig7)

