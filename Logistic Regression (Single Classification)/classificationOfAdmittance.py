# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 01:25:42 2020

@author: Jorge
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('./AdmittedBasedOnTest.csv')

twoTestScores = data.iloc[:,:2]
admitted = data.iloc[:, 2]

LogisticRegressionModel = LogisticRegression(random_state=0 , solver = 'lbfgs', multi_class='ovr').fit(twoTestScores,admitted)
LogisticRegressionModel.predict(twoTestScores.iloc[80:,:])
#this is how sure we are with this model
print(LogisticRegressionModel.score(twoTestScores,admitted))
#here is me pridicting if i got those scores and printing out the prediction
print(LogisticRegressionModel.predict([[80,50]]))