# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 23:10:00 2019

@author: HananeTech
"""
import numpy as np
from matplotlib import pyplot as plt

"""_______________________Compute one step of the gradient descent________________________"""
def gradient_descent(points, b, m, learning_rate):
    b_gradient = 0.0
    m_gradient = 0.0
    b_gradient -= np.mean(points[:, 1] -((m * points[:, 0]) + b))
    m_gradient -= np.mean(points[:, 0] * (points[:, 1] - ((m * points[:, 0]) + b)))
    new_b = b - (learning_rate*b_gradient)
    new_m = m - (learning_rate*m_gradient)
    return [new_b, new_m]

"""__________________________________Compute the error___________________________________"""
def cost(b, m, points):
    residuals = np.abs(points[:, 1] - (m*points[:, 0] + b))
    return np.mean(residuals)

"""__________________________________The training stage___________________________________"""
def linear_regression2D_training(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    error = []  #This list will contain the errors produced over time
    for i in range(num_iterations):
        [b, m] = gradient_descent(np.array(points), b, m, learning_rate)
        error.append(cost(b, m, np.array(points)))
            
    return b, m, error


"""______________________________Get the data from a csv file______________________________"""
points = np.genfromtxt("data.csv", delimiter=",")

"""___________________Initialize randomly the model coefficients b and m___________________"""
b = np.random.rand()
m = np.random.rand()
learning_rate = 0.0001
num_iterations = 1000
X1 = np.min(points[:,0])-10
X2 = np.max(points[:,0])+10
b, m, error = linear_regression2D_training(points, b, m, learning_rate, num_iterations)
Y1 = (m * X1) + b
Y2 = (m * X2) + b
plt.figure(1)
plt.plot([X1, X2], [Y1, Y2], color='red', label='Optimal line')
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()

plt.figure(2)
"""_______________Plot the errors produced during the first 100 iterations_______________"""
plt.plot(range(100), error[:100], color='red')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()