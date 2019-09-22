# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:38:07 2019

@author: HananeTech
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

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
    class_lengh = [N0, N1, N2]
    return dataset, targets, class_lengh

def LVQ_initialization(dataset, targets, class_lengh):
# =============================================================================
#     Initial Codebook: Contains two vectors of each class
#     labelbook: Contains the labels of the observations contained in Codebook
#     N: is the total number of observation
#     M: is the number of features
# =============================================================================
    N, M = dataset.shape
    codebook = []
    labelbook = []
    start = 0
    stop = 0
    for i,  Ni in enumerate(class_lengh):
        stop += Ni
        index = np.random.randint(start, stop, size=2)
        codebook.append(dataset[index, :])
        start += Ni
        labelbook.append(i)
        labelbook.append(i)
    
    codebook = np.reshape(np.array(codebook) , (2*len(class_lengh),M))   
    return codebook, labelbook
    
def update_codebook(codebook, labelbook, dataset, targets, LearningRate):
    N = len(dataset)
    l = len(codebook)
    for i in range(N):
        distances = np.zeros(l)  
        for j in range(l):
            distances[j] = np.linalg.norm(codebook[j]-dataset[i])
        index_ = np.argmin(distances)
        if labelbook[index_] == targets[i]:
            codebook[index_] = codebook[index_] + LearningRate*(dataset[i] - codebook[index_])
        else:
            codebook[index_] = codebook[index_] - LearningRate*(dataset[i] - codebook[index_])
    
    return codebook

def LVQ_training(dataset, targets, class_lengh, alpha, epochs):
    codebook, labelbook = LVQ_initialization(dataset, targets, class_lengh)
    for epoch in range(epochs):
        LearningRate = alpha*(1-(epoch/epochs))
        codebook = update_codebook(codebook, labelbook, dataset, targets, LearningRate)

    return codebook, labelbook

def LVQ_predicting(codebook, labelbook, datum):
    l, c = best_matching_unit.shape
    distances = np.zeros(l)  
    for i in range(l):
        distances[i] = np.linalg.norm(best_matching_unit[i] - datum)
    index_ = np.argmin(distances)
    predicted_class = labelbook[index_]
    
    return predicted_class

def plot_2D_training_data(dataset, targets, class_lengh, plt):
    start = 0
    for i, Ni in enumerate(class_lengh):
        labeli = 'Class ' + str(i)
        plt.scatter(dataset[start : start+Ni, 0], dataset[start : start+Ni, 1], c=targets[start : start+Ni], s= i*Ni+5, label=labeli)
        start += Ni
        

if __name__ == '__main__':
    """__________Get the data__________"""
    dataset, targets, class_lengh = generate_data()
    """__________Train the Model_______"""
    best_matching_unit, labelbook = LVQ_training(dataset, targets, class_lengh, alpha=0.3, epochs=50)
    
    """________Test the Model__________"""
    datum = [-5.0, 3.0]
    belongs_to = LVQ_predicting( best_matching_unit, labelbook, datum)
    print(datum, ' belongs to class ', belongs_to)
    
    plot_2D_training_data(dataset, targets, class_lengh, plt)
    """__Plot testing_datum in blue___"""
    plt.scatter(datum[0], datum[1], color='blue', label='Testing Data')
    style.use('ggplot')
    plt.grid(True, color='k')
    plt.legend()
    plt.show()
    