#time-based learning rate schedule
#learning rate decay over epochs
import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("ionosphere.csv", header = 0)
dataset = dataframe.values

X = dataset[:, 0:34].astype(float)
Y = dataset[:,34]

#one hot encoding
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

#create model
model = Sequential()
model.add(Dense(34, input_dim=34, init = 'normal', activation = 'relu'))
model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
epochs = 50
learning_rate = 0.1
decay_rate = learning_rate/epochs
momentum = 0.8

sgd = SGD(lr = learning_rate, momentum= momentum, decay = decay_rate, nesterov = False)
model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

model.fit(X, Y, validation_split=0.33, nb_epoch= epochs, batch_size = 28)