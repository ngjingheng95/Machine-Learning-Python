import numpy
from keras.models import Sequential
from keras.layers import Dense

#fix random seed
seed = 7
numpy.random.seed(seed)

#load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

#split into inputs (X) and outputs (Y)
X = dataset[:,0:8]
Y = dataset[:,8]

#create model (8 input neurons, 12 neurons in hidden layer 1,
#8 neurons in hidden layer 2, 1 output neuron with sigmoid function)
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#Compile model. 'loss' to evaluate set of weights,
#'optimizer' search through different weights for the network
#'metrics' used to report the classification accuracy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
#'nb_epoch' for number of epochs to run
#batch size => number of instances that are evaluated before a weight
# update in the network
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))