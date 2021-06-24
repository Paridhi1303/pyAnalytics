# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 20:10:09 2021

@author: rukap
"""

#standard libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree

url = url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/diabetes.csv'

data = pd.read_csv(url)
data.head()
data.columns
data.groupby('Outcome').aggregate({'Glucose':np.mean, 'BMI':np.mean})
data.groupby('Outcome').aggregate({'Glucose':np.mean, 'BMI':np.mean, 'BloodPressure':np.mean})
data.Outcome.value_counts()
data.shape
X= data.drop('Outcome', axis=1)
y=data['Outcome']
X
y

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) #comes from sklearn library
X_train.shape, X_test.shape
y_train.shape, y_test.shape

#make a decision tree model
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
help(clf)
#Train Decision Tree Classifier
clf=clf.fit(X_train, y_train)
y_train

#Test Model on test set data
y_pred=clf.predict(X_test)
y_pred
y_test

#Accuracy
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)
Accuracy=(117+43)/231
print("Accuracy", metrics.accuracy_score(y_test, y_pred))

#prune tree
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html
text_representation2 = tree.export_text(dtree)
print(text_representation2)
data.columns
text_representation3 = tree.export_text(dtree, feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'],  decimals=0, show_weights=True, max_depth=3)  #keep changing depth values
print(text_representation3)
