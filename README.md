# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import pandas, numpy, and matplotlib libraries.
2. Load the Placement Data CSV and display its info.
3. Drop 'sl_no' and 'salary' columns from the dataset.
4. Convert selected columns to categorical type and encode them numerically.
5. Separate features (x) and target (y), initialize random theta.
6. Define sigmoid, loss function, and gradient descent for training.
7. Train model using gradient descent and update theta.
8. Define predict function and make predictions on training data.
9. Calculate and print training accuracy.
10. Predict outcomes for two new input samples and print results.


## Program :
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Easwar R
RegisterNumber: 212223230053
*/
```
```
import pandas as pd
import numpy as np
data=pd.read_csv('Placement_data.csv')
data

data=data.drop('sl_no',axis=1)
data=data.drop('salary',axis=1)

data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data.dtypes

data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
y

theta=np.random.randn(x.shape[1])
Y=y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,Y):
    h=sigmoid(x.dot(theta))
    return -np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))

def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(Y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-Y)/m
        theta -= alpha*gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
    h=sigmoid(x.dot(theta))
    Y_pred=np.where(h >= 0.5,1,0)
    return Y_pred
Y_pred=predict(theta,x)

accuracy=np.mean(Y_pred.flatten()==Y)
print("Accuracy:",accuracy)

print(Y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
Y_prednew=predict(theta,xnew)
print(Y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
Y_prednew=predict(theta,xnew)
print(Y_prednew)

```

## Output :

### Preview Datasets :

![EX_6_OUTPUT_1](https://github.com/user-attachments/assets/c3cdc365-0520-40c4-a7f4-8cf83379a11c)

### Drop Sl.no & Salary && Display attribute type :

![EX_6_OUTPUT_2](https://github.com/user-attachments/assets/94b1b23f-f7d2-4e7c-857a-4a597b0d0dfd)


### Convertion of int to string:

![EX_6_OUTPUT_3](https://github.com/user-attachments/assets/b9d8906b-ff7a-490d-a849-860e0e1a2457)


### X-Initialize :

![EX_6_OUTPUT_4](https://github.com/user-attachments/assets/a8bebdd4-26db-438d-acf7-385ae84e81a9)


### Accuracy :

![EX_6_OUTPUT_5](https://github.com/user-attachments/assets/f41bcdfb-b769-46c2-8424-ffc105f2366a)

### Y_Predict :

![EX_6_OUTPUT_6](https://github.com/user-attachments/assets/b6e91620-a23f-424a-b544-f88e73524be4)


### Y :

![EX_6_OUTPUT_7](https://github.com/user-attachments/assets/c183dbdf-d124-455b-9f0b-f5328a33e053)


### Y_PredictNew :

![EX_6_OUTPUT_8](https://github.com/user-attachments/assets/7ff380b6-0333-49ef-9ccd-734228a6fd72)

![image](https://github.com/user-attachments/assets/0495587b-9e18-4134-94fa-08a885c18ffe)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
