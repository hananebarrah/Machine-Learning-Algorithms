# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:18:40 2019

@author: HananeTech
"""

import numpy as np
from matplotlib import pyplot as plt
 
def loss_function(X, Y, w):
    n = len(X[0])
    loss = 0.0
    for i, x in enumerate(X):
        f_x_y = (np.dot(x, w[:n]) + w[-1])*Y[i]
        if(f_x_y < 1):
            loss += 1 - f_x_y
    return loss
            

def svm_gd_train(X, Y):
    n = len(X[0])
    w  = np.zeros(n+1)
    eta = 1.0
    epochs = 755
    for epoch in range(1, epochs):
        for i in range(len(X)):
            if(Y[i]*((np.dot(X[i], w[:n]))+w[n]))<1:
                w[:n] = w[:n] + eta*(Y[i]*X[i] - (2.0/epoch)*w[:n])
                w[n] = w[n] + eta*(Y[i] - (2.0/epoch)*w[n])
            else:
                w = w - eta*(2.0/epoch)*w
        loss = loss_function(X, Y, w)
        print('Loss is: ', loss)
    return w

def svm_predict(x, w):
    y = 0
    n = len(x)
    if((np.dot(x, w[:n]) + w[-1])>= 1):
        y = 1
        plt.scatter(x[0],x[1], s=120, marker='+', linewidths=2, color='green')
    else:
        if ((np.dot(x, w[:n]) + w[-1])<= -1):
            y = -1
            plt.scatter(x[0],x[1], s=120, marker='_', linewidths=2, color='red')
    
    return y

def plot_data(X, Y):
    for i, x in enumerate(X):
        if Y[i] == 1:
            plt.scatter(x[0], x[1], s=120, marker='+', linewidths=2, color='green')
        else:
            plt.scatter(x[0], x[1], s=120, marker='_', linewidths=2, color='red')

def plot_hyperplane(w):
    # Plot the hyperplane calculated by svm_gd()
    x2=[w[0],w[1],-w[1],w[0]]
    x3=[w[0],w[1],w[1],-w[0]]

    x2x3 =np.array([x2,x3])
    X,Y,U,V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X,Y,U,V,scale=1, color='blue')
    plt.scatter(w[0],w[1], s=10, marker='o', linewidths=2, color='red')

if __name__ == '__main__':
    #Input data - Of the form [X value, Y value, Bias term]
    X = np.array([
        [-2,4],
        [4,1],
        [-1,2],
        [1, 6],
        [2, 4],
        [6, 2],
    ])
    
    #Associated output labels
    Y = np.array([-1,-1,-1,1,1,1])
    
    #Plot training data in a 2D graph!
    plot_data(X, Y)
    
    #Testing data
    x1 = [0, 1]
    x2 = [4, 4]

    #Train the model
    w = svm_gd_train(X, Y)
    
    #Plot the decision boundary
    plot_hyperplane(w)
    
    #Make predictions
    y1 = svm_predict(x1, w)
    y2 = svm_predict(x2, w)
    print('Y1 = ',y1)
    print('Y2 = ',y2)