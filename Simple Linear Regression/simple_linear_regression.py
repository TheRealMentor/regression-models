"""
Author: TheRealMentor
Created on: 19-07-2018

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting the Simple Linear Regression into the training set
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)

#Predicting from the test set
y_pred = clf.predict(X_test)

#Visualizing the model on training set
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, clf.predict(X_train), color="blue")
plt.title("Salary vs. Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary($)")
plt.show()

#Visualizing the model on test set
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, clf.predict(X_train), color="blue")
plt.title("Salary vs. Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary($)")
plt.show()