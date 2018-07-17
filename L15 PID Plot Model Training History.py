#import libraries
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy

seed = 7
numpy.random.seed(seed)

#load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:,0:8]
Y = dataset[:,8]

def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation= 'relu', init = 'uniform'))
    model.add(Dense(8, init = 'uniform', activation = 'relu'))
    model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return model

model = create_model()
estimator = model.fit(X, Y, batch_size=10, epochs = 150, validation_split=0.33, verbose = 0)
print(estimator.history.keys())

#summarize history for accuracy
#shows two plots: accuracy for train set and test set
plt.plot(estimator.history['acc'])
plt.plot(estimator.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

# summarize history for loss
plt.plot(estimator.history['loss'])
plt.plot(estimator.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()