#import libraries
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

seed = 7
numpy.random.seed(seed)

#load dataset, but keep only the top 5000 words
top_words = 5000
test_split = 0.33
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words = top_words, test_split = test_split)

# truncate reviews that are more than 500 words and zero-pad those that are shorter than 500 words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

def mlp_model():
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length = max_words))
    model.add(Flatten())
    model.add(Dense(250, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    return model

def cnn_model():
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length = max_words))
    model.add(Convolution1D(nb_filter = 32, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    return model

model = mlp_model()
model.fit(X_train, y_train, validation_data = (X_test, y_test), nb_epoch = 2, batch_size = 128, verbose = 1)
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1] * 100))


