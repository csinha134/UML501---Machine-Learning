#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 23:01:38 2022

@author: chayansinha
#Assignment 2 By Chayan Sinha 102003347
"""

#Q1
import pandas as pd
import numpy as np
# Q1(a)
df=pd.read_csv("/Users/chayansinha/Desktop/ML/archive/AWCustomers.csv")
#print(df)
new_df = df.iloc[:,[0,6,8,9,10,11,12,15,16,22]]
#print(new_df)
# Q1(b)
df1=pd.DataFrame(new_df)
print(df1)
#Q1(c)
df2 = df1.dtypes
print(df2)


#Q2(a)
print(df1.isna().sum())
df1.dropna();
print(df1);
#2(b)
#To implement Normaliztion , we have to implement Min Max Scaler 
print("Normalization of CustomerID and YearlyIncome")
from sklearn import preprocessing
df3=df1.iloc[:,[0,9]]
x=df3.values
min_max_scaler=preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(x)
df=pd.DataFrame(x_scaled)
print(df)

#2(c)
print("Discretization/Binning of Yearly Income")
print(pd.qcut(df1['YearlyIncome'],q=4))#qcut is for dicretization/binning

#2(d)
print("Standardization of CustomerID and YearlyIncome")
from sklearn.preprocessing import StandardScaler
from numpy import asarray
x1_data=df3.iloc[:,0].values
x2_data=df3.iloc[:,1].values
data= asarray([x1_data,x2_data]);
scaler=StandardScaler()
scaled_data=scaler.fit_transform(data)
print(scaled_data)

#2(e)
print("One Hot Encoding")
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# print(df1)
y=df1.iloc[:,2:5].values
ct= ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])],remainder='passthrough')
y=np.array(ct.fit_transform(y))
print(y)
############  label encoding on occupation
print(df1)
print("label Encoding on occupation")
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df1['Occupation']= label_encoder.fit_transform(df1['Occupation'])
print(df1['Occupation'].unique())

############# One Hot Encoding on occupation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
print("One Hot Encoding on occupation")
# print(df1)
y=df1.iloc[:,7:10].values
ct= ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
y=np.array(ct.fit_transform(y))
print(y)


#Q2 Manually using formula 
print("Normalization of CustomerID and YearlyIncome manually ")
from sklearn import preprocessing
df3 = df1.iloc[:,[0,9]]

# copy the data
df_min_max_scaled = df.copy()
  
# apply normalization techniques
for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
  
# view normalized data
print(df_min_max_scaled)
#######################################
print("Standardization of CustomerID and YearlyIncome")
new_d = df3.iloc[:,[0,1]]

# creating a Dataframe object
df = pd.DataFrame(new_d)
#print(df)
# Z-Score using pandas
df['CustomerID'] = (df['CustomerID'] - df['CustomerID'].mean()) / df['CustomerID'].std()
df['YearlyIncome'] = (df['YearlyIncome'] - df['YearlyIncome'].mean()) / df['YearlyIncome'].std()
print(df)


#Q3
dataset=pd.read_csv("/Users/chayansinha/Desktop/ML/archive/AWCustomers.csv")
dataset2=pd.read_csv("/Users/chayansinha/Desktop/ML/archive/AWSales.csv")

x=dataset2.iloc[:,2]
dataset.insert(23, "AvgMonthSpend",x, True)
new_df = dataset.iloc[:,[0,6,8,9,10,11,12,15,16,21,22,23]]
#print(new_df)
# Q1(b)
df5=pd.DataFrame(new_df)
print(df5.isna().sum())
df5.dropna()
print(df5)
from scipy.spatial import distance
#Q3(a)
print(distance.jaccard(df5['AvgMonthSpend'].values,df5['YearlyIncome'].values))
print(distance.cosine(df5['AvgMonthSpend'].values,df5['YearlyIncome'].values))
#Q3(b)
print(df5['AvgMonthSpend'].corr(df5['YearlyIncome']))
################## plotting correlation graph

# adds the title

import matplotlib.pyplot as plt
plt.title('Correlation Graph')
  

  
# plot the data
x=df5['AvgMonthSpend'].values
y=df5['YearlyIncome'].values
plt.scatter(x, y)
# # Labelling axes
plt.xlabel('Avg Month Spend')
plt.ylabel('Yearly Income')
plt.style.use('seaborn-white')
plt.show()
###### plotting correlation matrix
f = plt.figure(figsize=(7, 7))
plt.matshow(df5.corr(), fignum=f.number)
plt.xticks(range(df5.select_dtypes(['number']).shape[1]), df5.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df5.select_dtypes(['number']).shape[1]), df5.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)



