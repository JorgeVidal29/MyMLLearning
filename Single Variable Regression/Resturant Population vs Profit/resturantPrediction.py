
"""
Created on Mon May 18 21:56:49 2020

@author: Jorge
"""

#These are all the libs we are using 
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#reading the dataset and seperating it
dataset = pd.read_csv('./resturantData.csv')
population = dataset.iloc[:, :-1].values
profit = dataset.iloc[:, 1].values

#this splits the dataset into training and testing sets
populationTraining, populationTesting, profitTraining, profitTesting = train_test_split(population, profit, test_size=.2)

#this is the linear regression object of the sklearn lib
linearRegression = LinearRegression()

#this is the training on the training sets
linearRegression.fit(populationTraining, profitTraining)

# Make predictions using the testing set
profitPrediction = linearRegression.predict(populationTesting)

# The coefficients
print('Coefficients: \n', linearRegression.coef_)
# The mean squared error
print('Mean squared error: %.2f' % (metrics.mean_squared_error(profitTesting, profitPrediction)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % (metrics.r2_score(profitTesting, profitPrediction)))

#this is making the scatter plot for this data
plt.title('Population vs. Profit, Prediction Vs Actual')
plt.xlabel('Population in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.scatter(populationTesting,profitTesting, color = 'black')
plt.plot(populationTesting,profitPrediction, color = 'red')
plt.show()

