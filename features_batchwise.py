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
keys = np.array(D.keys())
   
D = np.array(D)
np.random.shuffle(D)

X = D[:, 0:14]
Y = D[:, 14:15]

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

start = 1991 
end = start + 6

keys_accuracies = np.zeros(keys.shape)

while(end <= 2018):
    X_batch = np.zeros((0, X.shape[1]))
    Y_batch = np.zeros((0, Y.shape[1]))
    
    for i, d in enumerate(D):
            yr = d[12]
            if yr >= start and yr <= end:
                X_batch = np.concatenate((X_batch, X[i:i+1, :]), axis = 0)
                Y_batch = np.concatenate((Y_batch, Y[i:i+1, :]), axis = 0)
                
    sz = X_batch.shape[0]
    
    cut = int(sz*3/4)
    
    X_train = X_batch[:cut, :]
    X_test = X_batch[cut:, :]
    
    Y_train = Y_batch[:cut, :]
    Y_test = Y_batch[cut:, :]
                
    for i in range(X_batch.shape[1]):        
        xf = X_train[:,i:i+1]
        yf = Y_train
        
        clf = LogisticRegression(random_state=0)
        clf.fit(xf, yf.ravel())
        
        xft = X_test[:,i:i+1]
        yft = Y_test
        #pft = clf.predict(xft)
        
        keys_accuracies[i]  += clf.score(xft, yft)
        
    start = end + 1
    end = start + 6
    
keys_accuracies /= 4

counts = []
for i, k in enumerate(keys):
    if i == 0 or i == 1 or i == 2 or i == 4 or i == 13:
        counts.append(keys_accuracies[i])
        
plt.bar(['danceability', 'energy', 'loudness', 'acousticness', 'Artist score'],counts, color = ['darkblue', 'lightblue','darkblue', 'lightblue', 'darkblue'])
plt.xlabel("Feature")
plt.ylabel("Accuracy")
plt.show()
            
        
