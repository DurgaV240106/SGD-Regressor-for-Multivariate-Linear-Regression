# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start Step
2.Data Preparation
3.Hypothesis Definition
4.Cost Function 
5.Parameter Update Rule 
6.Iterative Training 
7.Model Evaluation 
8.End

## Program:
```
/*

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: DURGA V
RegisterNumber: 212223230052

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
*/
```

## Output:
![image](https://github.com/user-attachments/assets/fa6280d3-c798-4614-b4e8-8c586488aa2d)

```
df.info()
```
## output:
![image](https://github.com/user-attachments/assets/6b7075b2-ab88-4de0-9dfa-904ef643a83e)
```
X=df.drop(columns=['AveOccup','target'])
X.info()`
```
## output:
![image](https://github.com/user-attachments/assets/09903a2e-381c-444f-accd-688bd735acdd)
```
Y=df[['AveOccup','target']]
Y.info()
```
## output:
![image](https://github.com/user-attachments/assets/5e7faccc-5298-436c-b41a-7bbb050ebfe3)
```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X.head()
```
## output:
![image](https://github.com/user-attachments/assets/5b410409-b658-49ee-87dc-f8f026151d27)
```
scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
```
## output:
![image](https://github.com/user-attachments/assets/d7535f93-5aa0-4313-8de8-6b69456c1029)

```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```
## output:
![image](https://github.com/user-attachments/assets/ffabb9c7-fae5-4439-b9a3-c1b8763dede1)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
