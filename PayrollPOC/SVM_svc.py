import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, metrics

#df = pd.read_csv("C:/ML.csv",header=None,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
payroll_results = pd.read_csv("C:/ML.csv",header=None)
payroll_results

def split_train_test(data, test_ratio):
    shuffled_indices =np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices] 

train_set, test_set = split_train_test(payroll_results, 0.2)
print(len(train_set)," train + ",len(test_set)," test")

train_split = np.split(train_set, [28], axis=1)
train_X = train_split[0]
train_Y = train_split[1]

classifier = svm.SVC(gamma=0.001)
classifier.fit(train_X,train_Y.values.ravel())

test_split = np.split(test_set, [28], axis=1)
test_X  = test_split[0]
predictions = classifier.predict(test_X)

predictions

test_Y = test_split[1]
print(test_Y.values.ravel())

classifier.score(test_X, test_Y.values.ravel())

testDataSet = pd.read_csv("C:/ML_test.csv",header=None)

newTestSet = np.split(testDataSet, [28], axis=1)

newTestSet[0]

classifier.predict(newTestSet[0])