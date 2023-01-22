# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#ASSIGNMENT 5
#NAME : CHAYAN SINHA
#ROLL NO : 102003347 
#Sub - Group 
#Q1
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
X.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.insert(X_scaled, 0, values=1, axis=1)
len(y[y == 0]), len(y[y == 1]), len(y[y == 2])


##Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
zero_list = []
one_list = []
for i in range(len(y_train)):
    if(y_train[i]==0):
        zero_list.append(i)
    elif(y_train[i]==1):
        one_list.append(i)
print(zero_list)
print(one_list)

len(X_train)

y_train_zero = []
y_train_one = []
for i in range(len(X_train)):
    if(zero_list.count(i)):
        y_train_zero.append(0)
    else:
        y_train_zero.append(1)
    if(one_list.count(i)):
        y_train_one.append(1)
    else:
        y_train_one.append(0)
print(y_train_zero)
print(y_train_one)

n = 1000
alpha = 0.01
m, k = X_train.shape
beta_zero = np.zeros(k)
for i in range(n):
    cost_gradient = np.zeros(k)
    z = X_train.dot(beta_zero)
    predicted = 1/(1+np.exp(-z))
    difference = predicted - y_train_zero
    for j in range(k):
        cost_gradient[j] = np.sum(difference.dot(X_train[:, j]))
    for j in range(k):
        beta_zero[j] = beta_zero[j] - (alpha/m)*cost_gradient[j]
print(beta_zero)

n = 1000
alpha = 0.01
m, k = X_train.shape
beta_one = np.zeros(k)
for i in range(n):
    cost_gradient = np.zeros(k)
    z = X_train.dot(beta_one)
    predicted = 1/(1+np.exp(-z))
    difference = predicted - y_train_one
    for j in range(k):
        cost_gradient[j] = np.sum(difference.dot(X_train[:, j]))
    for j in range(k):
        beta_one[j] = beta_one[j] - (alpha/m)*cost_gradient[j]
print(beta_one)

y_pred_zero = 1/(1+np.exp(-(X_test.dot(beta_zero))))
y_pred_one = 1/(1+np.exp(-(X_test.dot(beta_one))))

y_label_zero = np.zeros(len(y_pred_zero))
y_label_one = np.zeros(len(y_pred_one))

for i in range(len(y_pred_zero)):
    if(y_pred_zero[i]>=0.5):
        y_label_zero[i] = 1
    if(y_pred_one[i]>=0.5):
        y_label_one[i] = 1
print(y_label_one)
print(y_label_zero)

y_label = np.zeros(len(y_pred_zero))

for i in range(len(y_label)):
    if(y_label_zero[i]==1):
        if(y_label_one[i]==1):
            y_label[i] = 1
        else:
            y_label[i] = 2

print(y_label)

TP=0
TN=0
FP=0
FN=0 
y_test=np.array(y_test).reshape(-1,1) 
for i in range(len(y_label)):
    if(y_test[i]==1 and y_label[i]==1): 
        TP=TP+1
    if(y_test[i]==1 and y_label[i]==0): 
        FN=FN+1
    if(y_test[i]==0 and y_label[i]==1): 
        FP=FP+1
    if(y_test[i]==0 and y_label[i]==0): 
        TN=TN+1
print(TP,TN,FP,FN)
accuracy=(TP+TN)/(TP+TN+FP+FN)
print(accuracy*100)












#Q2 Ridge Logistic Regression
df=pd.read_csv('/Users/chayansinha/Library/Mobile Documents/com~apple~CloudDocs/SEM5/ML/ASSIGNMENT5/exam6.txt',
               names=['test1','test2','status'])
df.head()

X=df.drop('status',axis=1)
y=df['status']
X=np.array(X)
y=np.array(y)
print(X)
print(y)

##Importing PolyNomial Features
from sklearn.preprocessing import PolynomialFeatures
obj=PolynomialFeatures(degree=6)
X_transformed=obj.fit_transform(X)

X_transformed.shape

X_transformed

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, 
                                                    test_size=0.2, 
                                                    random_state=42)


##Part One  :  Step-by-Step Logistic Regression (with no regularization;
## alpha=10; number of iterations=1000)


n=1000
alpha=10
m,k=X_train.shape
beta=np.zeros(k)
for i in range(n):
    cost_gradient=np.zeros(k)
    z=X_train.dot(beta)
    predicted=1/(1+np.exp(-z))
    difference=predicted-y_train
    for j in range(k):
        cost_gradient[j]=np.sum(difference.dot(X_train[:,j]))
    for j in range(k):
        beta[j]=beta[j]-(alpha/m)*cost_gradient[j]
print(beta)
    
y_pred = 1/(1+np.exp(-(X_test.dot(beta))))
y_label = np.zeros(len(y_pred))
for i in range(len(y_pred)):
    if(y_pred[i]>=0.5):
        y_label[i] = 1
print(y_label)

##Calculating Accuracy using Confusion Matrix 
TP=0
TN=0
FP=0
FN=0
y_test=np.array(y_test).reshape(-1,1)
for i in range(len(y_label)):
    if(y_test[i]==1 and y_label[i]==1):
        TP=TP+1
    if(y_test[i]==1 and y_label[i]==0):
        FN=FN+1
    if(y_test[i]==0 and y_label[i]==1):
        FP=FP+1
    if(y_test[i]==0 and y_label[i]==0):
        TN=TN+1
print('TP:',TP,'FN:',FN)
print('FP:',FP,'TN:',TN)
print('Accuracy:',(TP+TN)*100/(TP+TN+FP+FN))

##Part Two Step-by-Step Logistic Regression 
#(with ridge regularization; alpha=10; number of iterations=1000; lambda=0.2)
n = 1000
alpha = 10
lam = 0.2
m, k = X_train.shape
beta = np.zeros(k)
for i in range(n):
    cost_gradient = np.zeros(k)
    z = X_train.dot(beta)
    predicted = 1/(1+np.exp(-z))
    difference = predicted - y_train
    for j in range(k):
        cost_gradient[j] = np.sum(difference.dot(X_train[:, j]))
    for j in range(k):
        beta[j] = beta[j]*(1-((alpha*lam)/m)) - (alpha/m)*cost_gradient[j]
print(beta)

##Finding the predicted the value
y_pred = 1/(1+np.exp(-(X_test.dot(beta))))
y_label = np.zeros(len(y_pred))
for i in range(len(y_pred)):
    if(y_pred[i]>=0.5):
        y_label[i] = 1
print(y_label)

##Calculating Accuracy using Confusion Matrix 
TP=0
TN=0
FP=0
FN=0
y_test=np.array(y_test).reshape(-1,1)
for i in range(len(y_label)):
    if(y_test[i]==1 and y_label[i]==1):
        TP=TP+1
    if(y_test[i]==1 and y_label[i]==0):
        FN=FN+1
    if(y_test[i]==0 and y_label[i]==1):
        FP=FP+1
    if(y_test[i]==0 and y_label[i]==0):
        TN=TN+1
print('TP:',TP,'FN:',FN)
print('FP:',FP,'TN:',TN)

print((TP+TN)*100/(TP+TN+FP+FN))






