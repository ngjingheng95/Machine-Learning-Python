#import libraries
from keras.models import Sequential
from keras.layers import Dense
import numpy
import os

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter = ",")

X = dataset[:,0:8]
Y = dataset[:,8]

def baseline_model():
    model = Sequential()
    model.add(Dense(12, input_dim = 8, init = 'uniform', activation = 'relu'))
    model.add(Dense(8, init = 'uniform', activation = 'relu'))
    model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(X, Y, nb_epoch = 100, batch_size= 10, verbose = 0)
scores = model.evaluate(X, Y, verbose = 0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#serialize model structure to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#serialize model weights to HDF5
model.save_weights("model.h5")
print("Saved!")