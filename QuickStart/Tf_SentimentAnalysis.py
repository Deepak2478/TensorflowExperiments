import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical

reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)

from collections import Counter
total_counts = Counter()
for _, row in reviews.iterrows():
    total_counts.update(row[0].split(' '))
#calculates the total words in a dataset    
print("Total words in data set: ", len(total_counts))
#Total words in data set:  74074


#Let's keep the first 10000 most frequent words.
# Most of the words in the vocabulary are rarely used so they will have little effect on our predictions.
# Below, we'll sort vocab by the count value and keep the 10000 most frequent words
vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]

print(vocab[:60])

#['', 'the', '.', 'and', 'a', 'of', 'to', 'is', 'br', 'it', 'in', 'i', 'this', 'that', 's', 'was', 'as',
# 'for', 'with', 'movie', 'but', 'film', 'you', 'on', 't', 'not', 'he', 'are', 'his', 'have', 'be', 'one'
# , 'all', 'at', 'they', 'by', 'an', 'who', 'so', 'from', 'like', 'there', 'her', 'or', 'just', 'about',
# 'out', 'if', 'has', 'what', 'some', 'good', 'can', 'more', 'she', 'when', 'very', 'up', 'time', 'no']

print(vocab[-1], ': ', total_counts[vocab[-1]])
#The last word in our vocabulary shows up in 30 reviews out of 25000. 
#I think it's fair to say this is a tiny proportion of reviews. We are probably fine with this number of words.

word2idx = {word: i for i, word in enumerate(vocab)}

def text_to_vector(text):
    word_vector = np.zeros(len(vocab), dtype=np.int_)
    for word in text.split(' '):
        idx = word2idx.get(word, None)
        if idx is None:
            continue
        else:
            word_vector[idx] += 1
    return np.array(word_vector)


text_to_vector('The tea is for a party to celebrate '
               'the movie so she has no time for a cake')[:65]
#array([0, 1, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0,
#       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
#       0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])



word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])
    
# Printing out the first 5 word vectors
word_vectors[:5, :23]
#array([[ 18,   9,  27,   1,   4,   4,   6,   4,   0,   2,   2,   5,   0,
#          4,   1,   0,   2,   0,   0,   0,   0,   0,   0],
#      [  5,   4,   8,   1,   7,   3,   1,   2,   0,   4,   0,   0,   0,
#          1,   2,   0,   0,   1,   3,   0,   0,   0,   1],
#       [ 78,  24,  12,   4,  17,   5,  20,   2,   8,   8,   2,   1,   1,
#          2,   8,   0,   5,   5,   4,   0,   2,   1,   4],
#       [167,  53,  23,   0,  22,  23,  13,  14,   8,  10,   8,  12,   9,
#          4,  11,   2,  11,   5,  11,   0,   5,   3,   0],
#       [ 19,  10,  11,   4,   6,   2,   2,   5,   0,   1,   2,   3,   1,
#          0,   0,   0,   3,   1,   0,   1,   0,   0,   0]])
Y = (labels=='positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)

trainY

# Network building
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    # Inputs
    net = tflearn.input_data([None, 10000])

    # Hidden layer(s)
    net = tflearn.fully_connected(net, 200, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')

    # Output layer
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', 
                             learning_rate=0.1, 
                             loss='categorical_crossentropy')
    
    model = tflearn.DNN(net)
    return model


model = build_model()

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=100)

predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)

# Helper function that uses your model to predict sentiment
def test_sentence(sentence):
    positive_prob = model.predict([text_to_vector(sentence.lower())])[0][1]
    print('Sentence: {}'.format(sentence))
    print('Positive' if positive_prob > 0.5 else 'Negative')

sentence = "Moonlight is by far the best movie of 2016."
test_sentence(sentence)

sentence = "It's amazing anyone could be talented enough to make something this spectacularly awful"
test_sentence(sentence)
