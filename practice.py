import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

balance_data = pd.read_csv('balance-scale.data.txt', sep= ',', header= None)
print "Dataset Lenght:: ", len(balance_data)
print "Dataset Shape:: ", balance_data.shape

print "Dataset:: "
balance_data.head()

X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
