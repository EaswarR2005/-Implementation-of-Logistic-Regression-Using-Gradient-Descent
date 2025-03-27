# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages 
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient. 
5. Define a function to plot the decision boundary and predict the Regression value


## Program & Output:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: EASWAR R
RegisterNumber: 212223230053
*/
```
```
import pandas as pd
import numpy as np
data=pd.read_csv('Placement_data.csv')
data
```

![image](https://github.com/user-attachments/assets/65dddcdb-3d69-494a-86de-b06e3d2bfb8a)

```
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
```

![image](https://github.com/user-attachments/assets/2cecab13-f7e3-4c66-a26f-d44192f12662)



```
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data
```

![ex6_op3](https://github.com/user-attachments/assets/cb85bec3-aed7-4e93-ab31-f963241e0afa)


```
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
y
```

![image](https://github.com/user-attachments/assets/bbc1513d-705c-400c-9a9b-e1bc5f27cf5b)


```
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
```

![image](https://github.com/user-attachments/assets/e520080a-db8c-40b7-814d-809dcc75e508)



```
print(Y_pred)
```

![image](https://github.com/user-attachments/assets/53bfa0c9-3e88-4f94-a771-c700ce376ab5)


```
print(Y)
```

![image](https://github.com/user-attachments/assets/eb6cbdd8-c9e7-4192-a89e-69bb9a4eb8a2)


```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
Y_prednew=predict(theta,xnew)
print(Y_prednew)
```


![ex6_op8](https://github.com/user-attachments/assets/f78aabae-94ef-4bd1-a76a-5f265824437a)



```
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
Y_prednew=predict(theta,xnew)
print(Y_prednew)
```


![ex6_op9](https://github.com/user-attachments/assets/ad5ae392-712c-4901-8146-071d6d6ed6cd)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
