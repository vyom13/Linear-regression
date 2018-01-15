
# coding: utf-8

# In[4]:

from sklearn import datasets
import pandas as pd
import numpy as np
from __future__ import division

iris = datasets.load_iris()
df = pd.DataFrame(np.concatenate((iris.data, np.array([iris.target]).T), axis=1), columns=iris.feature_names + ['target'])

df1 = df.drop(df.index[0:50])
df2 = df1.drop(['sepal length (cm)','sepal width (cm)'],axis = 1)

#Data setup

target = ['1', '2']
#Number of examples
m = df2.shape[0]
#Features
n = 2
#Number of classes
k = 2

X = np.ones((m,n + 1))
y = np.array((m,1))
X[:,1] = df2['petal length (cm)'].values
X[:,2] = df2['petal width (cm)'].values

#Labels
y = df2['target'].values
y=y-1

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
for j in range(1,3):
    X[:, j] = (X[:, j] - min(X[:,j]))/(max(X[:,j]) - min(X[:,j]))


# In[145]:

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def CostFunction(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = np.multiply(((-1)/m),(np.dot(np.transpose(y).reshape(1,100),(np.log(h))) + np.dot((np.transpose(1 - y).reshape(1,100)),(np.log(1 - h)))).item())
    return cost




def gradient(X,y,theta):
    m,n = X.shape
    h = sigmoid(np.dot(X,theta))
    delta = h - y.reshape(100,1)   
    gradient = np.dot(np.transpose(delta),X)
    gradient  = gradient/m
    return gradient

# In[146]:
    
def prediction(X,theta):
    h_predict = sigmoid(X.dot(theta))
    if (h_predict >= 0.5):
        return 1
    else:
        return 0 

total_error = 0

    
for k in range(100):
    X_test = X[k,:]
    X_train = np.delete(X,k,axis=0)
    Y_train = np.delete(y,k)
    Y_test = y[k]
    
    

    Lcost = []
    theta =  np.zeros((n + 1,1))
    for i in range(1000):
        L_rate = 0.50
        cost = CostFunction(theta,X,y)
        Lcost.append(cost)
        gr = gradient(X,y,theta)
        theta -= np.multiply(L_rate,gr.reshape(3,1))
        predictions = prediction(X_test,theta)
        error_each_iter = abs(Y_test - predictions)
        #error.append(error_each_fold)
        #print(error_each_fold)
    total_error = total_error + error_each_iter
Average_Error_Rate =     total_error/100.0
print('Average_Error_Rate',Average_Error_Rate)





# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



