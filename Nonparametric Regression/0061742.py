#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.array(pd.read_csv("hw04_data_set.csv", header=None))[1:].astype(float)
x_data = data[:, 0]
y_data = data[:, 1]

x_train = x_data[0:100]
y_train = y_data[0:100]

x_test = x_data[100:]
y_test = y_data[100:]


# In[2]:


def regressogram(x):
    bin_width = 3
    origin = 0    
    left = origin + int((x - origin) / bin_width) * bin_width
    right = origin + (int((x - origin) / bin_width) + 1) * bin_width
    values = y_train[np.where((x_train >= left) & (x_train < right))]
    sum_val = np.sum(values)
    return sum_val / len(values)

def running_mean_smoother(x):
    bin_width = 3
    left = (x - bin_width / 2)
    right = (x + bin_width / 2)
    values = y_train[np.where((x_train >= left) & (x_train <= right))]
    sum_val = np.sum(values)    
    return sum_val / len(values)

def kernel_smoother(x):
    bin_width = 1
    k = (x_train - x) / bin_width
    gauss_k = (1 / np.sqrt(2 * np.pi)) * np.exp(-(k * k) / 2)
    return np.dot(gauss_k, y_train) / np.sum(gauss_k)

def rmse(y_true, y_pred, length):
    return np.sqrt(np.sum((y_pred - y_true) * (y_pred - y_true)) / length)


# In[3]:


h=3
interval = np.arange(min(x_train) - h / 4, max(x_train) + h / 4, 0.001)
y_values = np.vectorize(regressogram)(interval)

plt.subplot()
plt.title("Regressogram")
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_train, y_train, alpha = 1, c = 'blue', edgecolors = 'none', label = "training")
plt.scatter(x_test, y_test, alpha = 1, c = 'red', edgecolors  = 'none', label = "test")
plt.plot(interval, y_values, c = 'black')
plt.legend(loc = 'best')
plt.show()

print("Regressogram => RMSE is ", rmse(np.vectorize(regressogram)(x_test), y_test, len(y_test)) , "when h is", h, ".")


# In[4]:


y_values = np.vectorize(running_mean_smoother)(interval)

plt.subplot()
plt.title("Running Mean Smoother")
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_train, y_train, alpha = 1, c = 'blue', edgecolors = 'none', label = "training")
plt.scatter(x_test, y_test, alpha = 1, c = 'red', edgecolors  = 'none', label = "test")
plt.plot(interval, y_values, c = 'black')
plt.legend(loc = 'best')
plt.show()

print("Running Mean Smoother => RMSE is ", rmse(np.vectorize(running_mean_smoother)(x_test), y_test, len(y_test)) , "when h is", h, ".")


# In[5]:


h=1
interval = np.arange(min(x_train) - h / 4, max(x_train) + h / 4, 0.001)
y_values = np.vectorize(kernel_smoother)(interval)

plt.subplot()
plt.title("Kernel Smoother")
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_train, y_train, alpha = 1, c = 'blue', edgecolors = 'none', label = "training")
plt.scatter(x_test, y_test, alpha = 1, c = 'red', edgecolors  = 'none', label = "test")
plt.plot(interval, y_values, c = 'black')
plt.legend(loc = 'best')
plt.show()

print("Kernel Smoother => RMSE is ", rmse(np.vectorize(kernel_smoother)(x_test), y_test, len(y_test)) , "when h is", h, ".")

