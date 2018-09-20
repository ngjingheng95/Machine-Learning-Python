import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)

#load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to become [samples][width][height][channels]
# channel is set to 1 because we are using grey scale image
# set channel to 3 for colour images (for the red, green and blue components)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

#normalize
X_train = X_train/255
X_test = X_test/255

#one hot encoding: convert the output into 10 columns with "0" and "1"
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
    model = Sequential()
    #visible layer with 28x28 inputs
    #convolutional layer with 32 feature maps, with the size of 5x5
    #border_mode = 'valid' means no padding around input or feature map
    #note that input_shape is <width>, <height>, <channels>
    model.add(Convolution2D(32, 5, 5, border_mode = 'valid', input_shape=(28, 28, 1), activation = 'relu'))
    #pooling layer that takes the maximum value
    model.add(MaxPooling2D(pool_size = (2, 2)))
    #dropout to reduce overfitting
    #randomly excludes 20% of neurons in the layer
    model.add(Dropout(0.2))
    #Flatten layer converts the 2D matrix data to a vector, allows the
    #output to be processed by standard fully connected layers
    model.add(Flatten())
    #hidden layer of 128 neurons
    model.add(Dense(128, activation = 'relu'))
    #output layer of 10 neurons
    model.add(Dense(num_classes, activation = 'softmax'))
    #compile model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def larger_model():
    model = Sequential()
    #30 feature maps, of size 5x5
    model.add(Convolution2D(30, 5, 5, border_mode = 'valid', input_shape = (28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Convolution2D(15, 3, 3, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.compile(loss = 'categorical_crossentropy')
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    #compile model, optimizer = 'adam', metrics = ['accuracy'])
    return model

model = larger_model()
model.fit(X_train, y_train, validation_data = (X_test, y_test), nb_epoch=10, batch_size = 200, verbose = 2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1]*100))


