# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:19:59 2019

@author: Aarush
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:00:27 2019

@author: Aarush
"""
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier

def preicsion_1(y, H):
    tp = 0
    pp = 0
    for i in range(len(H)):
        if H[i] == 1: 
            pp += 1
        if H[i] == 1 and y[i] == 1: 
            tp += 1
    if pp == 0:
        return 0
    return tp/pp
    
    
def preicsion_0(y, H):
    tp = 0
    pp = 0
    for i in range(len(H)):
        if H[i] == 0: 
            pp += 1
        if H[i] == 0 and y[i] == 0: 
            tp += 1
        if pp == 0:
            return 0
    return tp/pp

if __name__ == '__main__':
    
    D = pandas.read_csv('dataset_final.csv')
    
    D = np.array(D)
    np.random.shuffle(D)
        
    X = D[:, 0:5]
    X = np.concatenate(((X, D[:, 6:10])), axis = 1)
    X = np.concatenate(((X, D[:, 11:14])), axis = 1)
    Y = D[:, 14:15]
    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    f = 3
    sz = int(X.shape[0]/f)
    
    precision_0 = list()
    precision_1 = list()
    acc = list()
    recall = list()
    f1 = list()
    
    precision_0v = []
    precision_1v = list()
    accv = list()
    recallv = list()
    f1v = list()
    
    for i in range(f):
    
        train_set = np.zeros((0,X.shape[1]))
        train_y = np.zeros((0,1))
        
        for j in range(f):
            if j == i: 
                continue
            else:  
                train_set = np.concatenate((train_set,  X[j*sz: (j+1)*sz, :]),axis=0)
                train_y = np.concatenate((train_y, Y[j*sz: (j+1)*sz, :]), axis=0)
        
        clf = svm.SVC(kernel='rbf', gamma='scale', decision_function_shape='ovo')
        #clf = svm.SVC(kernel='poly', gamma='scale', degree = 2, decision_function_shape='ovo')
        #clf = svm.SVC(kernel='poly', gamma='scale', degree = 3, decision_function_shape='ovo')
        #clf = svm.SVC(kernel='linear', gamma='scale', decision_function_shape='ovo')
        
        clf.fit(train_set, train_y.ravel())
        h = clf.predict(train_set).ravel()
        a = train_y
    
        precision_1.append(preicsion_1(a, h))
        precision_0.append(preicsion_0(a, h))
        acc.append(accuracy_score(a, h))
        recall.append(recall_score(a,h))
        f1.append(f1_score(a,h))
           
        val_set = X[i*sz: (i+1)*sz, :]
        val_Y = Y[i*sz: (i+1)*sz, :]
        
        h = clf.predict(val_set).ravel()
        a = val_Y
        
        
        precision_1v.append(preicsion_0(a, h))
        precision_0v.append(preicsion_1(a, h))
        accv.append(accuracy_score(a, h))
        recallv.append(recall_score(a,h))
        f1v.append(f1_score(a,h))
        
    print("Accuracy\nPrecision(1)\nPrecision(0)\nRecall")
        
    print("TRAIN")
    print(mean(acc))
    print(mean(precision_1))
    print(mean(precision_0))
    print(mean(recall))
    #print(mean(f1))
    
    
    print("VAL")
    print(mean(accv))
    print(mean(precision_1v))
    print(mean(precision_0v))
    print(mean(recallv))
    #print(mean(f1v))
