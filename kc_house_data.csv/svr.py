# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:37:58 2023

@author: vijaya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\Notes\FSDS-PS\1.POLYNOMIAL REGRESSION- 18-03-2023\1.POLYNOMIAL REGRESSION\emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.svm import SVR
regressor = SVR(kernel="poly", degree = 4, gamma = 'auto')

regressor.fit(X, y)
y_pred = regressor.predict([[6.5]])

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
