# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 19:42:35 2021

@author: rukap
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as py


url= "https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/denco.csv"

df=pd.read_csv(url) #to import data

print(df)
df.shape #dimensions of data
df.columns
df.head(n=3)
len(df)
df.describe()
pd.options.display.float_format= '{:.2f}'.format

df['region'] = df['region'].astype('category')
#we can count and find mode etc for category
df.describe()

df.region.value_counts() #to see data for various regions and compare them
df.region.value_counts().plot(kind='bar') #to plot the data corresponding to various regions

#Q1 find customer-wise
df.custname.values_counts().sort_values(ascending=True).head(5)
df.groupby('custname').size()

#Q2 which customers contributed the most to their revenue
df.groupby('custname').revenue.sum().sort_values(ascending=False).head(5)
#another method
df.groupby('custname')['revenue'].aggregate([np.sum,max,min,'count']).sort_values(by='sum')

#Q3 What part numbers bring in to significant portion of revenue-maximise revenue from high value portion
df.groupby('partnum').revenue.aggregate([np.sum]).sort_values(by='sum', ascending=False).head(5)
df.groupby('partnum').revenue.aggregate([np.sum]).sort_values(by='sum', ascending=False).head(5).plot(kind='bar')

#maxprofit
df.groupby('partnum')['margin'].aggregate([np.sum]).sort_values(by='sum', ascending=False).head(5)

