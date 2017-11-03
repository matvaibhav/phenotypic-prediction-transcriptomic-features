import numpy as np
import os
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np

op = open("E:\Academics\CBProject\output_TPM.csv", "r")

dataset = np.loadtxt(op, delimiter=",")
print("Initial shape of dataset: ")
print(dataset.shape)

X = dataset[:, 0:-1]  # X is the list of the features for all the 369 rows
Y = dataset[:, -1]  # Y is the list of Labels for all the 369 rows (which range from 0-4)

n_samples = len(X)
trainPercent = 1
testPercent = 0
size_training = np.math.floor(n_samples * trainPercent)
size_test = np.math.floor(n_samples * testPercent)
decision_tree = DecisionTreeClassifier()
clfDT = decision_tree.fit(X[:size_training], Y[:size_training:])

expected = Y[size_test:]
predicted = decision_tree.predict(X[size_test:])

tree.export_graphviz(clfDT, out_file='tree.png')

# (clfDT.tree_.feature)  # gets array of nodes splitting feature

# Removing leaf nodes from the decision tree. The leaves have value=-2
b = np.array([-2])
c = np.setdiff1d(clfDT.tree_.feature, b)
print("Index of Reduced Features after Decision Tree", c)

X_decision_tree = dataset[:, c]
print("Size of data after feature reduction using decision tree : " + str(X_decision_tree.shape))

# SVM
trainPercent = 0.7
testPercent = 0.3
n_samples = len(X_decision_tree)
size_training = np.math.floor(n_samples * trainPercent)
size_test = np.math.floor(n_samples * testPercent)


# CROSS VALIDATION DT
scores = cross_val_score(clfDT, X_decision_tree, Y, cv=5)
print("Decision Tree 5 Fold Results:")
print("Scores:" +str(scores))
print("Mean:" +str(scores.mean()))

# CROSS VALIDATION SVM
svmclassifier = svm.SVC()
clfSVM = svmclassifier.fit(X_decision_tree[:size_training], Y[:size_training:])
expected = Y[size_test:]
predicted = clfSVM.predict(X_decision_tree[size_test:])
# print("Accuracy using SVM on the reduced feature set " + str(accuracy_score(expected, predicted)))

scores = cross_val_score(clfSVM, X_decision_tree, Y, cv=5)
print("SVM 5 Fold Results:")
print("Scores:" +str(scores))
print("Mean:" +str(scores.mean()))

# CROSS VALIDATION Random Forest
clfRF = RandomForestClassifier()
clfRF.fit(X_decision_tree[:size_training], Y[:size_training:])
expected = Y[size_test:]
predicted = clfRF.predict(X_decision_tree[size_test:])

scores = cross_val_score(clfRF, X_decision_tree, Y, cv=5)
print("Random Forest 5 Fold Results:")
print("Scores:" +str(scores))
print("Mean:" +str(scores.mean()))
