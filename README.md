# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Shrikrishna V
RegisterNumber:  212223040198
*/
```
```

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

## Placement Data:
![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/5c2110d2-f6cf-4aac-a2be-d95153bb7d19)
## Salary Data:
![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/13bb935e-8dae-483e-b652-ddbb70c121e2)
## Checking the null() function:
![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/6102728a-2948-40c6-96f5-6e298f70b89c)

## Data Duplicate:
![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/c5b70026-474d-4f44-a95d-4d4f0d2d8e60)

## Print Data:
![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/c7139247-c423-491d-a671-a7c04f10552b)

## Data-Status:
![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/d23026a5-619a-41ad-b306-9abb3c5b52b3)
## Y_prediction array:
![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/0af99585-c87c-4931-9040-346300c5f33b)

## Accuracy value:
![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/96651032-6ff0-4f08-ae06-1ecdf6b8dce0)

## Confusion array:

![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/c97bc103-ab90-4153-a4ea-9f00e8f057ca)
## Classification Report:
![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/51df3330-a49b-4786-a0a5-1d136a6d9a42)
## Prediction of LR:

![image](https://github.com/Wkrish28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144295230/553104c9-ab6b-4548-9c4c-28ec723d655d)









## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
