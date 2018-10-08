# Import Numpy, TensorFlow, TFLearn, and MNIST data
import numpy as np
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

# Retrieve the training and test data
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)
"""    
    One-hot encoding means writing categorical variables in a one-hot vector format, where the vector is all-zero apart from one element.
    4 can be represented like [0,1,0,0] 
"""

# Visualizing the data
import matplotlib.pyplot as plt
#matplotlib inline
    
# Display the first (index 0) training image
#display_digit(0)

# Define the neural network
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    # Inputs
    net = tflearn.input_data([None, trainX.shape[1]])

    """
     A rectified linear unit has output 0 if the input is less than 0, and raw output otherwise. 
     That is, if the input is greater than 0, the output is equal to the input. ReLUs' machinery is more like a real neuron in your body.
    """    
    # Hidden layer(s)
    net = tflearn.fully_connected(net, 128, activation='ReLU')
   
    net = tflearn.fully_connected(net, 32, activation='ReLU')
    
    # Output layer and training model
    """
    The softmax function squashes the outputs of each unit to be between 0 and 1, just like a sigmoid function. 
    But it also divides each output such that the total sum of the outputs is equal to 1
    """
    net = tflearn.fully_connected(net, 10, activation='softmax')
    
    """
    This parameter determines how fast or slow we will move towards the optimal weights. If the rate is very large we will skip the optimal solution. 
    If it is too small we will need too many iterations to converge to the best values.
    """
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')
    
    model = tflearn.DNN(net)
    return model

# Build the model
model = build_model()

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=100)
"""
    Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch
"""
# Compare the labels that our model predicts with the actual labels

# Find the indices of the most confident prediction for each item. That tells us the predicted digit for that sample.
predictions = np.array(model.predict(testX)).argmax(axis=1)

# Calculate the accuracy, which is the percentage of times the predicated labels matched the actual labels
actual = testY.argmax(axis=1)
test_accuracy = np.mean(predictions == actual, axis=0)

# Print out the result
print("Test accuracy: ", test_accuracy)

def test_digit(digit):
    arr=[]
    arr.append(digit)
    result_digit = np.array(model.predict(arr)).argmax(axis=1)
    print('Result digit:',result_digit[0])
    
# Function for displaying a training image by it's index in the MNIST set
def display_digit(index):
    label = testY[index].argmax(axis=0)
    # Reshape 784 array into 28x28 image
    image = testX[index].reshape([28,28])
    plt.title('Training data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()

# Call function
display_digit(6)
test_digit(testX[6]) 