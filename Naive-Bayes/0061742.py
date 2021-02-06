#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix

imagesdf = pd.read_csv('hw02_data_set_images.csv',header=None)
labeldf = pd.read_csv('hw02_data_set_labels.csv',header=None)


# In[2]:


# splitting the dataset
train_x1 = imagesdf.iloc[0:25]
train_x2 = imagesdf.iloc[39:64]
train_x3 = imagesdf.iloc[78:103]
train_x4 = imagesdf.iloc[117:142]
train_x5 = imagesdf.iloc[156:181]
train_x = pd.concat([train_x1, train_x2, train_x3, train_x4, train_x5])

test_x1 = imagesdf.iloc[25:39]
test_x2 = imagesdf.iloc[64:78]
test_x3 = imagesdf.iloc[103:117]
test_x4 = imagesdf.iloc[142:156]
test_x5 = imagesdf.iloc[181:195]
test_x = pd.concat([test_x1, test_x2, test_x3, test_x4, test_x5])

train_y1 = labeldf.iloc[0:25]
train_y2 = labeldf.iloc[39:64]
train_y3 = labeldf.iloc[78:103]
train_y4 = labeldf.iloc[117:142]
train_y5 = labeldf.iloc[156:181]
train_y = pd.concat([train_y1, train_y2, train_y3, train_y4, train_y5])

test_y1 = labeldf.iloc[25:39]
test_y2 = labeldf.iloc[64:78]
test_y3 = labeldf.iloc[103:117]
test_y4 = labeldf.iloc[142:156]
test_y5 = labeldf.iloc[181:195]
test_y = pd.concat([test_y1, test_y2, test_y3, test_y4, test_y5])


# In[3]:


# parameter estimation for a
a_means = (np.sum(train_x1, axis = 0)/len(train_y1))

a_stddev = np.sqrt((
        np.sum(train_x1 * train_x1, axis = 0)
        - 2 * np.sum(np.dot(train_x1, np.dot(a_means[:,np.newaxis], np.ones((1, len(a_means)))) * np.identity(len(a_means))), axis = 0) 
        + (len(train_y1) * (a_means * a_means)))
    /len(train_y1))

a_prior = len(train_y1)/len(train_y)


# In[4]:


# parameter estimation for b
b_means = (np.sum(train_x2, axis = 0)/len(train_y2))

b_stddev = np.sqrt((
        np.sum(train_x2 * train_x2, axis = 0)
        - 2 * np.sum(np.dot(train_x2, np.dot(b_means[:,np.newaxis], np.ones((1, len(b_means)))) * np.identity(len(b_means))), axis = 0) 
        + (len(train_y2) * (b_means * b_means)))
    /len(train_y2))

b_prior = len(train_y2)/len(train_y)


# In[5]:


# parameter estimation for c
c_means = (np.sum(train_x3, axis = 0)/len(train_y3))

c_stddev = np.sqrt((
        np.sum(train_x3 * train_x3, axis = 0)
        - 2 * np.sum(np.dot(train_x3, np.dot(c_means[:,np.newaxis], np.ones((1, len(c_means)))) * np.identity(len(c_means))), axis = 0) 
        + (len(train_y1) * (c_means * c_means)))
    /len(train_y3))

c_prior = len(train_y3)/len(train_y)


# In[6]:


# parameter estimation for d
d_means = (np.sum(train_x4, axis = 0)/len(train_y4))

d_stddev = np.sqrt((
        np.sum(train_x4 * train_x4, axis = 0)
        - 2 * np.sum(np.dot(train_x4, np.dot(d_means[:,np.newaxis], np.ones((1, len(d_means)))) * np.identity(len(d_means))), axis = 0) 
        + (len(train_y4) * (d_means * d_means)))
    /len(train_y4))

d_prior = len(train_y4)/len(train_y)


# In[7]:


# parameter estimation for e
e_means = (np.sum(train_x5, axis = 0)/len(train_y5))

e_stddev = np.sqrt((
        np.sum(train_x5 * train_x5, axis = 0)
        - 2 * np.sum(np.dot(train_x5, np.dot(e_means[:,np.newaxis], np.ones((1, len(e_means)))) * np.identity(len(e_means))), axis = 0) 
        + (len(train_y5) * (e_means * e_means)))
    /len(train_y5))

e_prior = len(train_y5)/len(train_y)


# In[8]:


# encoding the letters as integers
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


# In[9]:


# Naive-Bayes
def bayes(X, mean, stddev, prior):
    return np.sum(np.log((1/(stddev * np.sqrt(2 * np.pi))) * np.exp(((mean - X)*(X - mean))/(2 * stddev * stddev)))) + np.log(prior)


# In[10]:


# predictions
train_y_hat = train_x.apply(lambda x: np.argmax(np.array([bayes(x, a_means, a_stddev, a_prior), bayes(x, b_means, b_stddev, b_prior), bayes(x, c_means, c_stddev, c_prior), bayes(x, d_means, d_stddev, d_prior), bayes(x, e_means, e_stddev, e_prior)])), axis=1)
test_y_hat = test_x.apply(lambda x: np.argmax(np.array([bayes(x, a_means, a_stddev, a_prior), bayes(x, b_means, b_stddev, b_prior), bayes(x, c_means, c_stddev, c_prior), bayes(x, d_means, d_stddev, d_prior), bayes(x, e_means, e_stddev, e_prior)])), axis=1)


# In[11]:


print(confusion_matrix(train_y.tolist(), train_y_hat.tolist()))


# In[12]:


print(confusion_matrix(test_y.tolist(), test_y_hat.tolist()))


# In[23]:


import matplotlib.pyplot as plt

a = np.reshape(a_means.array, (16,20))
b = np.reshape(b_means.array, (16,20))
c = np.reshape(c_means.array, (16,20))
d = np.reshape(d_means.array, (16,20))
e = np.reshape(e_means.array, (16,20))


# In[30]:


plt.imshow(a.T, cmap='gray')


# In[31]:


plt.imshow(b.T, cmap='gray')


# In[32]:


plt.imshow(c.T, cmap='gray')


# In[33]:


plt.imshow(d.T, cmap='gray')


# In[34]:


plt.imshow(e.T, cmap='gray')
