import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)

#load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#shape[1] gives the image width and shape[2] gives the image height
#we define 'num_pixels' to get the length of array that we want to reshape into
#num_pixels = 28 * 28 = 784
num_pixels = X_train.shape[1] * X_train.shape[2]
#X_train is reshaped to become 60000 arrays with 784 pixel inputs
#pixel values are cast to float so we can normalize them easily
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
#normalize the pixel values to the range of 0 and 1 by dividing 255
#we divide by 255 as the pixel values are gray scale between 0 and 255
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs (transform the vector of class integers into a binary matrix)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
    model = Sequential()
    # 784 inputs, and 784 neurons in the hidden layer
    model.add(Dense(num_pixels, input_dim = num_pixels, init = 'normal', activation = 'relu'))
    #output layer with 10 class
    #use softmax activation function to turn outputs into probability-like values to select one class as the model's output
    model.add(Dense(num_classes, init = 'normal', activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                  metrics = ['accuracy'])
    return model

#building the model
model = baseline_model()
#fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch = 10, batch_size = 200, verbose = 2)
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

