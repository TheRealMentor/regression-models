"""
Author: TheRealMentor
Created on: 19-07-2018

"""

# Importing the libraries
import numpy as np
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding the categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, -1] = labelencoder.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:, 1:]

#Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting the multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)

#Predicting the test set
y_pred = clf.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
clf_OLS = sm.OLS(endog = y, exog = X_opt).fit()
clf_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
clf_OLS = sm.OLS(endog = y, exog = X_opt).fit()
clf_OLS.summary()

X_opt = X[:, [0,3,4,5]]
clf_OLS = sm.OLS(endog = y, exog = X_opt).fit()
clf_OLS.summary()

X_opt = X[:, [0,3,5]]
clf_OLS = sm.OLS(endog = y, exog = X_opt).fit()
clf_OLS.summary()

X_opt = X[:, [0,3]]
clf_OLS = sm.OLS(endog = y, exog = X_opt).fit()
clf_OLS.summary()
