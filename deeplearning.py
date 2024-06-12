# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:22:52 2020

@author: jacso
"""

import numpy as np

def activare(x, deriv = False):
    ''' sigmoid
    if(deriv == True):
        return x * (1 -x)
    return 1 / (1 + np.exp(-x))
    '''
    #''' tan
    if(deriv == True):
        return 1 -x ** 2
    return (np.exp(x) -np.exp(-x)) / (np.exp(x) + np.exp(-x))
    #'''
    ''' ReLU
    if(deriv == True and x >= 0):
        return 1
    if(x < 0):
        return 0
    return x
    '''
    #col = nr. neuroni, 1 este bias
X = np.array([[1, 0, 0, 1],
              [1, 0, 1, 1],
              [1, 1, 0, 1],
              [1, 1, 1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

#np.random.seed(1)
#seed-ul ne oferea aceleasi valori de fiecare data

#initializare ponderi (-1, 1)
neurons_in = 4
neurons_hidden = 5
neurons_out = 1
W0 = 2 * np.random.random((neurons_in, neurons_hidden)) -1
W1 = 2 * np.random.random((neurons_hidden, neurons_out))-1
alfa = 0.01#rata de invatare
for j in range(60000):
    #propagare inainte pt layers 0, 1 si 2
    l0 = X #nivel intrare
    l1 = activare(np.dot(l0, W0)) #nivel ascuns
    l2 = activare(np.dot(l1, W1))
    # evaluam eroarea dupa o trecere
    l2_error = y -l2
    
    if (j % 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))
        
    l2_delta = l2_error * activare(l2, deriv = True)
    l1_error = l2_delta.dot(W1.T)
    
    l1_delta = l1_error * activare(l1, deriv = True)
    #actualizam ponderile cu metoda gradient descent
    W1 += alfa * l1.T.dot(l2_delta)
    W0 += alfa * l0.T.dot(l1_delta)
print(l2)
    