# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 02:49:44 2019

@author: Priyanshi Jain
"""
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
start = 1991 
end = start + 6
accuracy_year = []
accuracy_val_year = []
while(end <= 2018):
    print("Start year : "+str(start))
    print("End year : "+str(end))
    
    X_batch = np.zeros((0, X.shape[1]))
    Y_batch = np.zeros((0, Y.shape[1]))
    
    for i, d in enumerate(D):
        yr = d[12]
        if yr >= start and yr <= end:
            X_batch = np.concatenate((X_batch, X[i:i+1, :]), axis = 0)
            Y_batch = np.concatenate((Y_batch, Y[i:i+1, :]), axis = 0)
    
    f = 3
    sz = int(X_batch.shape[0]/f)
    
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
    
        train_set = np.zeros((0,X_batch.shape[1]))
        train_y = np.zeros((0,1))
        
        for j in range(f):
            if j == i: 
                continue
            else:  
                train_set = np.concatenate((train_set,  X_batch[j*sz: (j+1)*sz, :]),axis=0)
                train_y = np.concatenate((train_y, Y_batch[j*sz: (j+1)*sz, :]), axis=0)
        
        clf = RandomForestClassifier(n_estimators=100,criterion = "entropy", max_depth=i, random_state=0)
    
        clf.fit(train_set, train_y.ravel())
        h = clf.predict(train_set).ravel()
        a = train_y
        
        acc.append(accuracy_score(a, h))

           
        val_set = X[i*sz: (i+1)*sz, :]
        val_Y = Y[i*sz: (i+1)*sz, :]
        
        h = clf.predict(val_set).ravel()
        a = val_Y
        
        accv.append(accuracy_score(a, h))
        
    
        
    print("TRAIN")
    accuracy_year.append(np.mean(acc))
    
    
    print("VAL")
    accuracy_val_year.append(np.mean(accv))
    
    start = end + 1
    end = start + 6  
    
    
svm_val = [0.77, 0.73, 0.73, 0.77]
accuracy_val_year = [0.53,0.62,0.61,0.64]
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
x = np.arange(4)
width = 0.35

svm = ax.bar(x-width/2,svm_val, label = 'SVM', width = 0.35)
rf = ax.bar(x+width/2,accuracy_val_year, label = 'Random Forests', width = 0.35)
 
ax.set_xticks(range(4))
ax.set_xticklabels(['1991-1997','1998-2004','2005-2011','2012-2018']) 

ax.legend()

for rect in rf:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom') 
for rect in svm:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), 
                textcoords="offset points",
                ha='center', va='bottom') 

fig.tight_layout()

plt.show()

