# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:04:59 2019

@author: Aarush
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
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
from sklearn.neural_network import MLPClassifier


D = pandas.read_csv('dataset_final.csv')
print(D.keys())
keys = np.array(D.keys())
   
D = np.array(D)
np.random.shuffle(D)

X = D[:, 0:14]
Y = D[:, 14:15]

X_train = X[:3000, :]
Y_train = Y[:3000, :]

X_test = X[3000:, :]
Y_test = Y[3000:, :]

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

counts = []

for i in range(X.shape[1]):
    
    print(keys[i])
    
    xf = X_train[:,i:i+1]
    yf = Y_train
    
    clf = LogisticRegression(random_state=0)
    clf.fit(xf, yf.ravel())
    
    xft = X_test[:,i:i+1]
    yft = Y_test
    pft = clf.predict(xft)
    
    acc = accuracy_score(yft, pft)
    
    if i == 0 or i == 1 or i == 2 or i == 4 or i == 13:
        counts.append(acc)
    
    
plt.bar(['danceability', 'energy', 'loudness', 'acousticness', 'Artist score'],counts, color = ['darkblue', 'lightblue','darkblue', 'lightblue', 'darkblue'])
plt.xlabel("Feature")
plt.ylabel("Accuracy")
plt.show()





    
