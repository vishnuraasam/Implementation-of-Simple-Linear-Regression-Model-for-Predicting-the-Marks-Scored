# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
~~~
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rasam Vishnu
RegisterNumber:  212220040131
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("/content/sample_data/student_scores.csv")
dataset.head()
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn .linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
LinearRegression()
Y_pred=regressor.predict(X_test)
plt.scatter(X_train,Y_train,color='yellow')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title("h vs s(training set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
plt.scatter(X_test,Y_test,color='blue')
plt.plot(X_test,regressor.predict(X_test),color='green')
plt.title("h vs s(training set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
dataset.tail()
~~~

## Output:

![output1](https://user-images.githubusercontent.com/103240414/165911123-7fd62442-22d9-434c-9160-2c5ce921811b.png)

![output2](https://user-images.githubusercontent.com/103240414/165911160-1b414771-1fc4-4362-b78c-6c77a394d902.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
