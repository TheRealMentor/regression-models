# Polynomial Regression
"""
Author: TheRealMentor
Created on: 05-08-2018

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""
No spiltting of dataset because of less number of inputs.
Whole dataset is used to train the model.

"""

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_poly, y)

#For higher resolution curve
X_grid = np.arange(min(X), max(X)+0.1, 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

#Visualising the polynomial regression result
plt.scatter(X, y, c="red")
plt.plot(X_grid, clf.predict(poly_reg.fit_transform(X_grid)), c="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

#Predicting the salary for a given number of experience years
clf.predict(poly_reg.fit_transform(6.5))