import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import tensorflow as tf
#import tflearn

payroll_results=pd.read_csv("C:/ML_cleaned.csv")
payroll_classifier=pd.read_csv("C:/Payroll_classifier.csv")
print(payroll_results)
#payroll_classifier.head()

print(payroll_results.head())

print(payroll_results.describe())

payroll_results.hist()
plt.show()

def split_train_test(data, test_ratio):
    shuffled_indices =np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]  

train_X, test_X = split_train_test(payroll_results, 0.2)
print(len(train_X)," train + ",len(test_X)," test")

train_Y, test_Y = split_train_test(payroll_classifier, 0.2)
print(len(train_Y)," train + ",len(test_Y)," test")


#def build_model():
#    tf.reset_default_graph()
    
#    net = tflearn.input_data([None, 10000])

#    net = tflearn.fully_connected(net, 200, activation='ReLU')
#    net = tflearn.fully_connected(net, 25, activation='ReLU')

#    net = tflearn.fully_connected(net, 2, activation='softmax')
#    net = tflearn.regression(net, optimizer='sgd', 
#                             learning_rate=0.1, 
#                            loss='categorical_crossentropy')
    
#    model = tflearn.DNN(net)
#    return model


#model = build_model()

# Training
#model.fit(train_X, train_Y, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=100)
#print('model trained')
