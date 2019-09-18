# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:05:49 2019

@author: HananeTech
"""

import numpy as np
from matplotlib import pyplot as plt

def generate_data():
    N0 = 10     #Number of observations in the first class
    N1 = 15     #Number of observations in the second class
    class0 = np.random.randn(N0, 2)
    target0 = np.zeros(N0)
    class1 = np.random.randn(N1, 2) + np.array([7, 2])
    target1 = np.ones(N1)
    data = np.concatenate((class0, class1))
    targets = np.concatenate((target0, target1))
    return data, targets

def pdf(x, mean, sigma2):
    return (1.0/np.sqrt(2*np.pi*sigma2))*np.exp(-((x-mean)**2)/(2*sigma2))

def statistical_properties(data, targets):
    N = len(targets)       #Total number of observations 
    D = len(data[0, :])    #Number of features (attribues)
    classes_set = list(set(targets))
    K = len(classes_set)    #Number of classes
    
    means = np.zeros((K, D))
    sigma2 = np.zeros((K, D))
    nk = np.zeros(K)        #Number of observations in each class
    for k in range(K):
        for i in range(N):
            if(targets[i] == classes_set[k]):
                means[k] += data[i]
                nk[k] +=1
        means[k] = means[k]/nk[k]
        
    for k in range(K):
        for i in range(N):
            if(targets[i] == classes_set[k]):
                sigma2[k] += (data[i]-means[k])**2
        sigma2[k] = sigma2[k] / nk[k]
    
    return means, sigma2, nk, N, K

def class_probabilities(nk, N, K):
    c_probabilities = nk / N
    return c_probabilities

def gaussian_naive_bayes_prediction(datum, means, sigma2, c_probabilities):
    P_X_C = np.ones(len(c_probabilities))  #Probabilities of belongingness of datum to all the classes
    for k in range(len(c_probabilities)):
        tmps = pdf(datum, means[k], sigma2[k])
        for tmp in tmps:
           P_X_C[k] = P_X_C[k]*tmp
           
    P_X_C = P_X_C*c_probabilities
    
    return np.argmax(P_X_C)
    
if __name__ == '__main__':
    """__________Get the data__________"""
    data, targets = generate_data()
    
    """_______Learn the Model__________"""
    means, sigma2, nk, N, K = statistical_properties(data, targets)
    c_probabilities = class_probabilities(nk, N, K)
    
    """________Test the Model__________"""
    test_datum = [7.0, 2.0]
    belongs_to = gaussian_naive_bayes_prediction(test_datum, means, sigma2, c_probabilities)
    print(test_datum, ' belongs to class ', belongs_to)
    plt.scatter(data[:, 0], data[:, 1], c=targets)
    
    """__Plot testing_datum in blue___"""
    plt.scatter(test_datum[0], test_datum[1], color='blue')