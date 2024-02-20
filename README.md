# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the neccessary libraries 
2. Read the csv file using pd.read_csv
3. Seperate the independent from the dependent values
4. Split the data
5. Create a regression model
6. Find the MSE , MAE, RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: RUCHITRA THIYAGARAJ
RegisterNumber:212223110043
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
print(df.head())
print(df.tail())

x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)

print(y_test)
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("HOURS vs SCORES (Training set)")
plt.xlabel("HOURS")
plt.ylabel("SCORES")
plt.show()

plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="blue")
plt.title("HOURS vs SCORES (Test set)")
plt.xlabel("HOURS")
plt.ylabel("SCORES")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("rmse = ",rmse)
*/
```

## Output:
![image](https://github.com/RuchitraThiyagaraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/154776996/b9eaf8a1-aacf-41d1-8705-130d21a5f2d1)
![image](https://github.com/RuchitraThiyagaraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/154776996/bbe0d4a8-057a-4040-ab82-bee5b0196ba2)
![image](https://github.com/RuchitraThiyagaraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/154776996/04e7060c-2a02-4b45-9528-cb1c7ea70b51)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
