# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import standard libraries in python for finding Decision tree classsifier model for predicting employee churn.

2.Initialize and print the Data.head(),data.info(),data.isnull().sum()

3. Visualize data value count.

4. Import sklearn from LabelEncoder.

5. Split data into training and testing.

6. Calculate the accuracy, data prediction by importing the required modules from sklearn

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: G.R.DHANUSH
RegisterNumber:  212221040038
*/
```
import pandas as pd

data=pd.read_csv("/content/Employee.csv")

print("data.head():")

data.head()

print("data.info():")

data.info()

print("isnull() and sum():")

data.isnull().sum()

print("data value counts():")

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

print("data.head() for Salary:")

data["salary"]=le.fit_transform(data["salary"])

data.head()

print("x.head():")

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()

y=data["left"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

print("Accuracy value:")

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy

print("Data Prediction:")

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

## Output:
![image](https://github.com/Dhanush12022004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135558/1d8bbb73-8cfd-4f39-9504-2aacb2390527)

![image](https://github.com/Dhanush12022004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135558/b94b902d-500a-4345-af05-ca5f7e25bf61)

![image](https://github.com/Dhanush12022004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135558/9ce1c1f6-f843-4133-a48d-23cfc6affc2b)

![image](https://github.com/Dhanush12022004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135558/9a290f2b-dbdb-4430-8365-fb3e5d2a26c3)


![image](https://github.com/Dhanush12022004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135558/e9d27e76-60f2-482f-a21f-e538518e33a0)


![image](https://github.com/Dhanush12022004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135558/16bdfd0d-e8b6-4a04-85dd-049eb92f8468)

![image](https://github.com/Dhanush12022004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135558/7f9975be-67b6-4eb6-a7b8-c50676af8e23)

![image](https://github.com/Dhanush12022004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135558/90580e73-baea-4ec2-9cdb-2c8dadf1266d)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
