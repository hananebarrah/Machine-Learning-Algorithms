# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:56:55 2019

@author: HananeTech
"""

import numpy as np
from matplotlib import pyplot as plt

def generate_data():
    N0 = 7      #Number of observations in the first class
    N1 = 10     #Number of observations in the second class
    N2 = 15     #Number of observations in the third class
    class0 = np.random.randn(N0, 2)
    target0 = np.zeros(N0)
    class1 = 1.5*np.random.randn(N1, 2) + np.array([7, 2])
    target1 = np.ones(N1)
    class2 = 0.8*np.random.randn(N2, 2) + np.array([-4, 4])
    target2 = 2*np.ones(N2)
    dataset = np.concatenate((np.concatenate((class0, class1)), class2))
    targets = np.concatenate((np.concatenate((target0, target1)), target2))
    return dataset, targets, N0 + N1 + N2

def KNN_predicting(dataset, targets, N, datum, k):
    distances = np.zeros(N)  
    for i in range(N):
        distances[i] = np.linalg.norm(dataset[i] - datum)
    index_k = (np.argsort(distances))[:k]
    list_k = np.array(targets[index_k], dtype='int32')
    predicted_class = np.bincount(list_k).argmax()
    
    return predicted_class


if __name__ == '__main__':
    """__________Get the data__________"""
    dataset, targets, N = generate_data()
    
    """________Test the Model__________"""
    datum = [-4.0, 3.0]
    belongs_to = KNN_predicting(dataset, targets, N, datum, k=3)
    print(datum, ' belongs to class ', belongs_to)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=targets)
    
    """__Plot testing_datum in blue___"""
    plt.scatter(datum[0], datum[1], color='blue')