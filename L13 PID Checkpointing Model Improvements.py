## contains 2 checkpoint functions:
## - checkpoint_every_improvement(): creates a new file every time there is improvement
## - checkpoint_best(): only 1 file is created, which contains the best performing weights at the end of the training

#import libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:, 0:8]
Y = dataset[:,8]

def baseline_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, init = 'uniform', activation = 'relu'))
    model.add(Dense(8, init = 'uniform', activation = 'relu'))
    model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def checkpoint_every_improvement():
    #creates a new hdf5 file everytime there is an improvement
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only= True, mode= 'max')
    callbacks_list = [checkpoint]
    return callbacks_list

def checkpoint_best():
    #creates only 1 file with the best performing set of weights
    filepath = "weights-best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only= True, mode= 'max')
    callbacks_list = [checkpoint]
    return callbacks_list

model = baseline_model()
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10, callbacks=checkpoint_best(), verbose = 0)