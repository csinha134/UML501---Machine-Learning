#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:10:40 2022

@author: chayansinha
"""
#Chayan Sinha 102003347 3COE14

#Q3
import numpy as np;
import pandas as pd;
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder,Imputer
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import metrics

#a.
df=pd.read_csv("/Users/chayansinha/Desktop/imports-85.csv")
print(df)
df.columns=["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
            "num_doors", "body_style", "drive_wheels", "engine_location", 
            "wheel_base", "length", "width", "height", "curb_weight",
            "engine_type", "num_cylinders", "engine_size", "fuel_system"
            , "bore", "stroke", "compression_ratio", "horsepower"
            , "peak_rpm", "city_mpg", "highway_mpg", "price"]



print(df)

#b.
df.isnull().sum()

filter=df =="?"
df.where(filter).count()

#df['normalized_losses'].replace(to_replace='?',value=NAN,inplace=True)
#So we have Normalised Losses,num_doors,bore,stroke,horsepower,peak_rpm,price;

#normalised_losses
rep=df[df['normalized_losses']!='?']
normalized_losses_median=rep['normalized_losses'].median();
df['normalized_losses'].replace(to_replace="?",value=normalized_losses_median,
                                inplace=True)
print(df['normalized_losses'])

#num_doors
print(df['num_doors'])
rep=df[df['num_doors']!='?']
rep=rep[rep['aspiration']=='turbo']
num_doors_mode=rep['num_doors'].mode();
print(num_doors_mode)
df['num_doors'].replace(to_replace="?",value="four",
                                inplace=True)
print(df['num_doors'])

#price
rep=df[df['price']!='?']
price_median=rep['price'].median();
print(price_median)
df['price'].replace(to_replace='?',value=price_median,inplace=True)
print(df['price'])






