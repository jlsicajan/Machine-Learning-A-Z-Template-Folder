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

#Encoding categorial data (independent variable)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

column_trasformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
independent_variables = np.array(column_trasformer.fit_transform(independent_variables))

#Encoding categorial data (dependent variable)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dependent_variables = label_encoder.fit_transform(dependent_variables)

#Feature Scaling (Standardisation)
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
independent_variables = standard_scaler.fit_transform(independent_variables)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
dependent_train, dependent_test, independent_train, independent_test = train_test_split(dependent_variables, independent_variables, test_size=0.2, random_state=0)
