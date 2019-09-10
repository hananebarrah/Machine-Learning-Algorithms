# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:15:03 2019

@author: HananeTech
"""
import numpy as np
from matplotlib import pyplot as plt

"""__________________________Generate Synthetic data___________________________"""
def generate_data():
    N = [20, 20]        #Number of elements in each class
    class1 = np.random.normal(4, 1, N[0])
    y1 = np.zeros(class1.shape)     #Labels of class1
    class2 = np.random.normal(20, 1, N[1])
    y2 = np.ones(class2.shape)      #Labels of class2
    data = np.concatenate((class1, class2))
    y = np.concatenate((y1, y2))
    return data, y, N

"""____________________________Learning the Model______________________________"""
def LDA_model_learning(data, N):
    """N is a vector that contains the number of elements in each class"""
    K = len(N)
    meank = np.zeros(K)
    probk = np.zeros(K)
    """______Compute the mean of each class_______"""
    start = 0
    for k in range(K):
        meank[k] = np.average(data[start:start+N[k]])
        start += N[k]
        
    """______Compute the Class Probabilities______"""
    probk = np.array(N)/len(data)
    """___________Compute  the Variance___________"""
    sigma = 0
    start = 0
    for k in range(K):
        sigma += np.sum((data[start : start+N[k]] - meank[k])**2)
        start += N[k]
        
    sigma = sigma * (1.0 / (len(data)-K))
    return meank, probk, sigma

"""_____________________________Prediction Stage_______________________________"""
def LDA_prediction(meanK, probK, sigma, datum):
    discriminant = datum*(meanK/sigma) - (meanK**2/(2*sigma)) + np.log(probK)
    print(discriminant[0])
    return np.argmax(discriminant)
  
"""__________________________Showing the data in 2D____________________________"""
def show_result(data, y):
    plt.scatter(data, y, c=y, label='Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Discriminant Analysis')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    data, y, N = generate_data()
    meanK, probK, sigma = LDA_model_learning(data, N)
    print("The means are ", meanK)
    print("Probability of each class", probK)
    print("The Variance is ",  sigma)
    """______________________________Testing Stage_____________________________"""
    datum = 5.24
    belongsTo = LDA_prediction(meanK, probK, sigma, datum)
    print(datum, " belong to ", belongsTo)
    show_result(data, y)