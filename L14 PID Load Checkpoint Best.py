## load best performing weights

#import libraries
from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 7
numpy.random.seed(seed)

def load_model(filename):
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation = 'relu'))
    model.add(Dense(8, init='uniform', activation = 'relu'))
    model.add(Dense(1, init='uniform', activation = 'sigmoid'))
    model.load_weights(filename)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

model = load_model("weights-best.hdf5")
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
