# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 00:59:14 2019

@author: HananeTech
"""

import numpy as np
import matplotlib.pyplot as plt

def get_data():
    """_______________________Nummbers of observation per cluster________________________"""
    obs_per_class = 50
    
    X1 = np.random.randn(obs_per_class, 2) + np.array([3, 3])
    X2 = np.random.randn(obs_per_class, 2) + np.array([-6, -6])
    X3 = np.random.randn(obs_per_class, 2) + np.array([6, -6])
    data = np.concatenate((np.concatenate((X1, X2)), X3))
    return data

def euclidien_dist(X1, X2):
    """______________The Euclidien distance between 2 data points X1 and X2______________"""
    return np.linalg.norm(X1 - X2)

def k_means(data, k, epsilon=0):
    """________________________________k is the clusters number________________________________
       _____________________m is the data length, n is the features number_____________________
       ____________________________epsilon is the converging error_____________________________"""
    [m, n] = data.shape
    
    """_____datum_class is column vector where we put the belonging class of each datum_______"""
    datum_class = np.zeros((m, 1))
    
    """____Initialize the clusters centers with k random data points from the input dataset___"""
    centers = data[np.random.randint(0, m-1, size=k)]
    
    """dist_norm is the distance between two consecutive centers ||new_centers- old_centers||_"""
    dist_norm = 1   
    iteration = 0
    while dist_norm > epsilon:
        iteration += 1
        old_centers = np.copy(centers)
        for index, datum in enumerate(data):
            dist = []
            for j, center in enumerate(centers):
                dist.append(euclidien_dist(datum, center))
            
            """________Get the position of the smallest distance________"""
            datum_class[index, 0] = np.argmin(dist) 
        
        for j in range(k):
            cluster_j = [i for i in range(len(datum_class)) if datum_class[i] == j]
            centers[j, :] = np.mean(data[cluster_j], axis=0)
            
        dist_norm = euclidien_dist(centers, old_centers)
        print(dist_norm)
    
    return centers, datum_class


data = get_data()

centers, datum_class = k_means(data, 3)
print(centers)
plt.scatter(data[:, 0], data[:, 1], s=30, c=datum_class[:, 0])
plt.scatter(centers[:, 0], centers[:, 1], c='b')