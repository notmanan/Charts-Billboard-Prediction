# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:36:01 2019

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


    
D = pandas.read_csv('dataset_final.csv')
print(D.keys())
D = np.array(D)
np.random.shuffle(D)
    
X = D[:, 0:5]
X = np.concatenate(((X, D[:, 6:10])), axis = 1)
X = np.concatenate(((X, D[:, 11:14])), axis = 1)
Y = D[:, 14:15]

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

x_pos = list()
y_pos = list()

x_neg = list()
y_neg = list()

for i in range(D.shape[0]):
    if Y[i] == 1:
        x_pos.append(D[i][0])
        y_pos.append(D[i][4])
    else:
        x_neg.append(D[i][0])
        y_neg.append(D[i][4])

plt.cla()
plt.scatter(x_neg, y_neg, color = 'red' , s = 3, label = 'Non billboard')        
plt.scatter(x_pos, y_pos, color = 'blue', s = 3, label = 'Billboard')
plt.legend(loc = 'top right')
plt.xlabel('danceability')
plt.ylabel('acousticness')

x_pos = list()
y_pos = list()

x_neg = list()
y_neg = list()

for i in range(D.shape[0]):
    if Y[i] == 1:
        x_pos.append(D[i][1])
        y_pos.append(D[i][11])
    else:
        x_neg.append(D[i][1])
        y_neg.append(D[i][11])

plt.cla()
plt.scatter(x_neg, y_neg, color = 'red' , s = 3, label = 'Non billboard')        
plt.scatter(x_pos, y_pos, color = 'blue', s = 3, label = 'Billboard')
plt.legend(loc = 'lower left')
plt.xlabel('energy')
plt.ylabel('artist popularity')

def plot_boundary(X, Y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
    
    
    clf = svm.SVC(kernel='linear', gamma='scale', decision_function_shape='ovo')
    clf.fit(X, Y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    
    plt.cla()
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
X1 = D[:, 1:2]
X1 = np.concatenate((X1, D[:, 11:12]), axis = 1)
plot_boundary(X1, Y)


