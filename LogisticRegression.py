# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:55:24 2019

@author: HananeTech
"""

import numpy as np
import matplotlib.pyplot as plt


"""__________________________Generate Synthetic data___________________________"""
def get_data():
    # Nummbers of observation per class
    obs_per_class = 50
    
    X1 = np.random.randn(obs_per_class, 2) + np.array([-2, -2])
    X2 = np.random.randn(obs_per_class, 2) + np.array([2, 2])
    X = np.concatenate((X1, X2))
    y = np.concatenate((np.zeros(obs_per_class), np.ones(obs_per_class)))
    return X, y

"""____________________________The sigmoid function____________________________"""
def sigmoid(X, B, bias):
    return 1/(1+np.exp(-bias - (B.dot(X))))

"""____________Logistic Resgession with Stochastic Gradient Descent____________"""
def logistic_regression_sgd(X, y, epochs, l_rate):
    n = len(X[0]) #Number of features
    B = np.zeros(n)
    bias = 0
    errors = []   #This vector will contain the errors performed in each epoch
    for epoch in range(epochs):
        err = np.zeros(y.shape)
        for i in range(len(X)):
            predict = sigmoid(X[i, :], B, bias)
            err[i] = (predict - y[i])**2
            B = B + (l_rate*(y[i]-predict)*predict*(1-predict)*X[i])
            bias = bias + (l_rate*(y[i]-predict)*predict*(1-predict))
        errors.append(np.mean(err))
    return B, bias, errors

def to_predict(x, B, bias):
    return  np.round(sigmoid(x, B, bias))


X, y = get_data()

epochs = 40     #Iterations number
l_rate = 0.1    #Learning rate
N = len(X)      #Observations number
n = len(X[0])   #Features number

x_new = np.array([[2, 1]])    #Data to predict after the training process

"""_____________________________The training stage_____________________________"""
B_final, bias_final, errors = logistic_regression_sgd(X, y, epochs, l_rate)

"""_______________________Print the learned coefficients_______________________"""
print(B_final,"\n", bias_final)
print("_________X1___________________________X2__________________Prediction_________")
for i in range(N):
    print(X[i, 0]," ||______|| ", X[i, 1], ": ", sigmoid(X[i], B_final, bias_final))

""""________________________Prediction of a new data x_________________________"""
prediction = to_predict(x_new[0], B_final, bias_final)
print(x_new, "belons to the group labeled with ", prediction)
print("Errors:\n", errors)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], s=30, c=y)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Training dataset after the learning process")

"""_________________Draw the line that separates the 2 classes_________________"""
x_axis = np.linspace(-5, 5, 2)
y_axis = -(B_final[0]/B_final[1])*x_axis - (bias_final/B_final[1])
plt.plot(x_axis, y_axis, color="red")
plt.show()

plt.figure(2)
plt.plot(range(epochs), errors, color="red")
plt.xlabel("Epochs")
plt.ylabel("Errors")
plt.show()