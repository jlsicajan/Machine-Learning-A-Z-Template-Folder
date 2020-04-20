# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 03:11:51 2020

@author: jlsicajan

Data Preprocessing

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import datasets
datasets = pd.read_csv('Data.csv')

independent_variables = datasets.iloc[:, :-1].values
dependent_variables = datasets.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(independent_variables[:, 1:3])

independent_variables[:, 1:3] = imputer.transform(independent_variables[:, 1:3])

# They will replace the NaN values by the average in the column


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
independent_variables[:, 0] = labelencoder_x.fit_transform(independent_variables[:, 0])
onehotencoder = OneHotEncoder()

#Encoding categorial data\
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

column_trasformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
independent_variables = np.array(column_trasformer.fit_transform(independent_variables))
