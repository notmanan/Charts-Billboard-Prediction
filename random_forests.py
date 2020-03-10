# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:25:34 2019

@author: Priyanshi Jain
"""

import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, accuracy_score


D = pandas.read_csv('dataset_final.csv')
print(D.keys())
D = np.array(D)

np.random.shuffle(D)

X = D[:, 0:5]
Y = D[:,14:15]

X = np.concatenate(((X, D[:, 6:9])), axis = 1)
X = np.concatenate(((X, D[:, 11:12])), axis = 1)


ohe = OneHotEncoder()
X = np.concatenate(((X,ohe.fit_transform(D[:,12:13]).toarray())),axis=1)

X = np.concatenate(((X, D[:, 13:14])), axis = 1)

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

folds = 3
size = int(X.shape[0]/folds)

def precision_1(y, H):
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
    
def precision_0(y, H):
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

accuracy_train = []
accuracy_test = []

precision_1_train = []
precision_0_train = []

precision_1_test = []
precision_0_test = []

recall_1_train = []
recall_0_train = []

recall_1_test = []
recall_0_test = []


for i in range(folds):
    
    X_train = np.zeros((0,X.shape[1]))
    Y_train = np.zeros((0,1))
    
    for j in range(folds):
        if j == i: 
            continue
        else:  
            X_train = np.concatenate((X_train,  X[j*size: (j+1)*size, :]),axis=0)
            Y_train = np.concatenate((Y_train, Y[j*size: (j+1)*size, :]), axis=0)
    val_set = X[i*size: (i+1)*size, :]
    val_Y = Y[i*size: (i+1)*size, :]
    clf = RandomForestClassifier(n_estimators=100,criterion = "entropy", max_depth=6, random_state=0)
    clf.fit(X_train, Y_train)
    accuracy_train.append(clf.score(X_train,Y_train.ravel()))
    accuracy_test.append(clf.score(val_set,val_Y.ravel()))
    
    precision_1_train.append(precision_1(Y_train,clf.predict(X_train)))
    precision_1_test.append(precision_1(val_Y,clf.predict(val_set)))
    
    precision_0_train.append(precision_0(Y_train,clf.predict(X_train)))
    precision_0_test.append(precision_0(val_Y,clf.predict(val_set)))
    
    recall_1_train.append(recall_score(Y_train,clf.predict(X_train)))
    recall_1_test.append(recall_score(val_Y,clf.predict(val_set)))
    
    recall_0_train.append(recall_score(Y_train,clf.predict(X_train)))
    recall_0_test.append(recall_score(val_Y,clf.predict(val_set)))
    
print("Random Forest")
print("Accuracy on Train Set: ", np.mean(accuracy_train))
print("Accuracy on Test Set: ", np.mean(accuracy_test))
print("Precision on Train Set: (of 0 class) ", np.mean(precision_0_train))
print("Precision on Train Set: (of 1 class) ", np.mean(precision_1_train))
print("Precision on Test Set: (of 0 class) ", np.mean(precision_0_test))
print("Precision on Test Set: (of 1 class) ", np.mean(precision_1_test))

print("Recall on Train Set: (of 0 class) ", np.mean(recall_0_train))
print("Recall on Train Set: (of 1 class) ", np.mean(recall_1_train))
print("Recall on Test Set: (of 0 class) ", np.mean(recall_0_test))
print("Recall on Test Set: (of 1 class) ", np.mean(recall_1_test))


Y = Y.astype(int)
X_train = X[:3150,:]
Y_train = Y[:3150,:]

X_test = X[3150:,:]
Y_test = Y[3150:,:]



accuracyvsdepth = []
accuracy_train = []

for i in range(3,15):
    clf = RandomForestClassifier(n_estimators=100,criterion = "entropy", max_depth=i, random_state=0)
    clf.fit(X_train, Y_train)
    accuracyvsdepth.append(clf.score(X_test,Y_test))
    accuracy_train.append(clf.score(X_train,Y_train))
    
import matplotlib.pyplot as plt

plt.plot(range(3,15),accuracyvsdepth, label = 'Test')
plt.plot(range(3,15),accuracy_train,color = 'red', label = 'Train')
plt.legend()
plt.xlabel("Max depth of the tree")
plt.ylabel("Accuracy")
plt.title('Random forests')
plt.show()
    

