import pandas
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LearningRateScheduler

#takes in epoch as argument and returns the learning rate
#this decay function computes the amount of drop in learning rate after a time interval = epochs drop
def step_decay(epoch):
    initial_lrate = 0.1
    #how much to drop the learning rate by, in this case we half the learning rate every x epochs
    drop = 0.5
    #learning rate drops after how many epochs?
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

seed = 7
numpy.random.seed(seed)
# header = 0 as first row is the header in the csv file
#header = none if there is no header in csv file
dataframe = pandas.read_csv("ionosphere.csv", header = 0)
dataset = dataframe.values

X = dataset[:, 0:34].astype(float)
Y = dataset[:,34]

#one hot encoding
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

# create model
model = Sequential()
model.add(Dense(34, input_dim=34, init = 'normal', activation = 'relu'))
model.add(Dense(1, init = 'normal', activation = 'sigmoid'))

sgd = SGD(lr = 0.0, momentum = 0.9, decay=0.0, nesterov=False)
model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

model.fit(X, Y, validation_split=0.33, nb_epoch=50, batch_size=28, callbacks=callbacks_list)


