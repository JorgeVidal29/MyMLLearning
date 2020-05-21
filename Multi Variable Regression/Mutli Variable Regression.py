#Multiple Linear Regression
"""
Created on Thur May 21 17:47:49 2020

@author: Jorge
"""

#These are all the libs we are using 
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#reading the dataset and seperating it
dataset = pd.read_csv('./housePrices.csv')
"The x contains two columns the first is the square feet of the house and the second is the the number of bedrooms of the house"
xSet = dataset.iloc[:, :-1].values
price = dataset.iloc[:, 2].values

#this splits the dataset into training and testing sets
xTraining, xTesting, priceTraining, priceTesting = train_test_split(xSet, price, test_size=.2)

#this is the linear regression object of the sklearn lib
linearRegression = LinearRegression()

#this is the training on the training sets
linearRegression.fit(xTraining, priceTraining)

# Make predictions using the testing set
pricePrediction = linearRegression.predict(xTesting)

# The coefficients
print('Coefficients: \n', linearRegression.coef_)
# The mean squared error
print('Mean squared error: %.2f' % (metrics.mean_squared_error(priceTesting, pricePrediction)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % (metrics.r2_score(priceTesting, pricePrediction)))


