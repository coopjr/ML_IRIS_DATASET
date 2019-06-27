# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:24:54 2019

@author: Sandeep
"""
#IRIS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('IRIS.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

