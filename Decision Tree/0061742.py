#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rmse(y_true, y_pred, length):
    return np.sqrt(np.sum((y_pred - y_true) * (y_pred - y_true)) / length)

class DecisionTree:
    def __init__(self, p, data, label):        
        self.p = p
        self.data = data
        self.label = label
        self.size = data.shape[0]   
        self.terminal = (self.size <= self.p)
        
        if self.size > self.p:
            self.median = (np.unique(self.data)[1:] + np.unique(self.data)[:-1]) / 2
        
        self.error = ((self.label - self.getAverage()) ** 2).sum() / self.size

    def getAverage(self):
        self.average = self.label.sum() / self.data.shape[0]
        return self.average

    def getError(self):
        return ((self.label - self.getAverage()) ** 2).sum() / self.size

    def splitChildren(self, weight):
        left_child = DecisionTree(self.p, self.data[self.data <= weight], self.label[self.data <= weight])
        right_child = DecisionTree(self.p, self.data[self.data > weight], self.label[self.data > weight])
        return (left_child.size * left_child.getError() + right_child.size * right_child.getError()) / self.size

    def getSplit(self):
        if self.size > self.p:           
            self.ind = self.median[np.vectorize(self.splitChildren)(self.median).argmin()]
            self.c_true = DecisionTree(self.p, self.data[self.data <= self.ind], self.label[self.data <= self.ind])
            self.c_false = DecisionTree(self.p, self.data[self.data > self.ind], self.label[self.data > self.ind])
            self.c_true.getSplit()
            self.c_false.getSplit()

    def getScore(self, x_test):
        if self.terminal:
            return self.average
        if x_test <= self.ind:
            return self.c_true.getScore(x_test)
        else:
            return self.c_false.getScore(x_test)
        
###################################################################################################################
        
data = pd.read_csv('hw05_data_set.csv')
X = data['x'].values
Y = data['y'].values
    
x_train = X[:100]
x_test = X[100:]
y_train = Y[:100]    
y_test = Y[100:]
    
p = 15
       
decisionTree = DecisionTree(p, x_train, y_train)

decisionTree.getSplit()

interval = np.linspace(0, 60, 1000)
y_values = np.vectorize(decisionTree.getScore)(interval)

plt.title("P = 15")
plt.plot(interval, y_values, 'black')
plt.scatter(x_train, y_train, alpha = 1, c = 'blue', edgecolors = 'none', label = "training")
plt.scatter(x_test, y_test, alpha = 1, c = 'red', edgecolors = 'none', label = "test")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'best')
plt.plot(interval, y_values, 'black')
plt.show()

y_pred = np.vectorize(decisionTree.getScore)(x_test)

print('RMSE is', rmse(y_test, y_pred, len(y_test)), "when P is", p)

###################################################################################################################

set_error = []

for p in np.arange(5, 55, 5):
    set_tree = DecisionTree(p, x_train, y_train)
    set_tree.getSplit()
    
    set_pred = np.vectorize(set_tree.getScore)(x_test)
    set_error.append(rmse(y_test, set_pred, len(y_test)))

plt.plot(np.arange(5, 55, 5), set_error, '--ok')
plt.xlabel('Pre-Pruning size (P)')
plt.ylabel('RMSE')
plt.show()


