# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:04:13 2019

@author: HananeTech
"""

import numpy as np

def extract_list(l):
    new_l =[]
    for el in l:
        if el not in new_l:
            new_l.append(el)
    return new_l

def get_targets(data):
    return extract_list(data[:,-1])            

def get_domain(data, index):    
    return extract_list(data[:,index])

def countCj(data, cj):
    return list(data[:,-1]).count(cj)

def countXiCj(data, xi, cj, index):
    p = 0.0
    for i in range(len(data)):
        if((data[i, index]==xi) and (data[i, -1]==cj)):
            p +=1.0
    return p

def class_probabilities(data, targets):
    prob = np.zeros((len(targets), 1))
    for i, target in enumerate(targets):
        prob[i] = countCj(data, target)/len(data)
    return prob

def conditional_probabilities(data, targets):
    nbrAtt = len(data[0,:])     #Number of attributes
    domains = []                #Domains values
    P_Xi_Cj = []
    for index in range(nbrAtt-1):
        domain = get_domain(data, index)
        for xi in domain:
            P_xi_cj = []
            for cj in targets:
                P_xi_cj.append((countXiCj(data, xi, cj, index)/countCj(data, cj)))
            P_Xi_Cj.append(P_xi_cj)
        
        domains=np.concatenate((domains, domain))
    
    return P_Xi_Cj, domains
    
def print_probabilities(targets, domains, P_Xi_Cj):
    for i, xi in enumerate(domains):
        pxc=P_Xi_Cj[i]
        for j, cj in enumerate(targets):
            print('P({}|{})={}'.format(xi, cj, pxc[j]))

def naive_bayes_prediction(P_Cj, P_Xi_Cj, domains, test_data):
    probList = []
    for j in range(len(P_Cj)) :
        prob = 1.0
        for i, attribute in enumerate(test_data):
            print(attribute)
            i = list(domains).index(attribute)
            prob *= (P_Xi_Cj[i])[j]
        prob = prob*P_Cj[j]
        probList.append(prob)
    return np.argmax(probList), np.max(probList)
        
    
if __name__ == '__main__':
    """__________Get the data__________"""
    dt=np.dtype('U25')
    data = np.genfromtxt("naive_bayes_data.csv", delimiter=";", dtype=dt)
    targets = get_targets(data)
    P_Xi_Cj, domains = conditional_probabilities(data, targets)
    test_data = ['"sunny', 'working']
    class_index, belongProba = naive_bayes_prediction(class_probabilities(data, targets), P_Xi_Cj, domains, test_data)
    print('Targets are: ', targets)
    print('Probabilities: ', P_Xi_Cj)
    print('Domains are: ', domains)
    print_probabilities(targets, domains, P_Xi_Cj)
    print(test_data,' belongs to ', targets[class_index], ' with probability of ', belongProba)