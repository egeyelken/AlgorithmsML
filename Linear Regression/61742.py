#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder


# In[2]:


imagesdf = pd.read_csv('hw02_data_set_images.csv',header=None)
labeldf = pd.read_csv('hw02_data_set_labels.csv',header=None)


# In[3]:


train_x1 = imagesdf.iloc[0:25]
train_x2 = imagesdf.iloc[39:64]
train_x3 = imagesdf.iloc[78:103]
train_x4 = imagesdf.iloc[117:142]
train_x5 = imagesdf.iloc[156:181]
train_x = pd.concat([train_x1, train_x2, train_x3, train_x4, train_x5])


# In[4]:


test_x1 = imagesdf.iloc[25:39]
test_x2 = imagesdf.iloc[64:78]
test_x3 = imagesdf.iloc[103:117]
test_x4 = imagesdf.iloc[142:156]
test_x5 = imagesdf.iloc[181:195]
test_x = pd.concat([test_x1, test_x2, test_x3, test_x4, test_x5])


# In[5]:


train_y1 = labeldf.iloc[0:25]
train_y2 = labeldf.iloc[39:64]
train_y3 = labeldf.iloc[78:103]
train_y4 = labeldf.iloc[117:142]
train_y5 = labeldf.iloc[156:181]
train_y = pd.concat([train_y1, train_y2, train_y3, train_y4, train_y5])


# In[6]:


test_y1 = labeldf.iloc[25:39]
test_y2 = labeldf.iloc[64:78]
test_y3 = labeldf.iloc[103:117]
test_y4 = labeldf.iloc[142:156]
test_y5 = labeldf.iloc[181:195]
test_y = pd.concat([test_y1, test_y2, test_y3, test_y4, test_y5])


# In[7]:


def sigmoid(X,W,w0):
    sigmoid = (1/(1 + np.exp(-1*(np.dot(X,W)+w0))))
    return sigmoid

def update_W(X,y_truth,y_pred):
    a = (y_truth - y_pred) * y_pred * (1 - y_pred)
    return (np.dot(-1*(np.transpose(X)),a))
def update_w0(y_truth,y_pred):
    a =((y_truth - y_pred) * y_pred * (1- y_pred))
    col_sums = a.sum(axis = 0)
    return -1 * col_sums

eta = 0.01
epsilon = 1e-3

np.random.seed(421)
W = np.random.uniform(low = -0.01, high = 0.01, size = (train_x.shape[1], 5))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, 5))


# In[8]:


onehotencoder = OneHotEncoder(categories = 'auto')

y_train_array = np.asarray(train_y).reshape(125,1)
y_test_array = np.asarray(test_y).reshape(70,1)

encoded_y_train = onehotencoder.fit_transform(train_y).toarray()
encoded_y_test = onehotencoder.fit_transform(test_y).toarray()


# In[9]:


objective_values = []
iteration = 1
while 1:
    y_predicted = sigmoid(train_x,W,w0)
    objective_values.append(np.sum((encoded_y_train - y_predicted)**2)*(0.5))

    prew_W = W
    W = W - eta * update_W(train_x,encoded_y_train,y_predicted)
    
    prew_wo = w0
    w0 = w0 - eta*update_w0(encoded_y_train,y_predicted)
    
    q = np.sqrt(np.sum((w0 - prew_wo)**2) + np.sum((W - prew_W)**2))
     
    if(iteration >= 500):
        break
    
    if(epsilon > q):
        break
       
    iteration += 1

y_predicted.shape


# In[10]:


import matplotlib.pyplot as plt
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show


# In[11]:


objective_values1 = []
iteration1 = 1
while 1:
    y_predicted1 = sigmoid(test_x,W,w0)
    objective_values.append(np.sum((encoded_y_test - y_predicted1)**2)*(0.5))

    prew_W = W
    W = W - eta * update_W(test_x,encoded_y_test,y_predicted1)
    
    prew_wo = w0
    w0 = w0 - eta*update_w0(encoded_y_test,y_predicted1)
    
    q = np.sqrt(np.sum((w0 - prew_wo)**2) + np.sum((W - prew_W)**2))
    
   
    if(iteration >= 500):
        break
    
    if(epsilon > q):
        break
    
    
    iteration += 1


# In[12]:


y_predicted = 1 * (y_predicted > 0.5)
y_predicted1 = 1 * (y_predicted1 > 0.5)

train_y = np.where(train_y == 'A', 0, train_y) 
train_y = np.where(train_y == 'B', 1, train_y) 
train_y = np.where(train_y == 'C', 2, train_y) 
train_y = np.where(train_y == 'D', 3, train_y) 
train_y = np.where(train_y == 'E', 4, train_y) 

test_y = np.where(test_y == 'A', 0, test_y) 
test_y = np.where(test_y == 'B', 1, test_y) 
test_y = np.where(test_y == 'C', 2, test_y) 
test_y = np.where(test_y == 'D', 3, test_y) 
test_y = np.where(test_y == 'E', 4, test_y) 

encoded_y_train = onehotencoder.fit_transform(train_y).toarray().astype(int)
encoded_y_test = onehotencoder.fit_transform(test_y).toarray().astype(int)

train_confusion_matrix = confusion_matrix(y_predicted.argmax(axis=1), encoded_y_train.argmax(axis=1))
test_confusion_matrix = confusion_matrix(y_predicted1.argmax(axis=1), encoded_y_test.argmax(axis=1))


print(train_confusion_matrix)
print('\n')
print(test_confusion_matrix)