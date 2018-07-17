#a total of 3 networks are created here
#1. baseline model
#2. dropout at input layer
#3. dropout at both hidden layers

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

seed = 7
numpy.random.seed(seed)

#load dataset
dataframe = pandas.read_csv("sonar.csv", header = None)
dataset = dataframe.values
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

#one hot encoding
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def create_baseline():
    model = Sequential()
    model.add(Dense(60, input_dim = 60, init = 'normal', activation = 'relu'))
    model.add(Dense(30, init = 'normal', activation = 'relu'))
    model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
    # using stochastic gradient descent optimizer
    # 'lr' is learning rate, 'momentum' dampens oscillations and accelerates SGD in relevant direction, 'decay' is learning rate decay
    sgd = SGD(lr = 0.01, momentum = 0.8, decay = 0.0, nesterov = False)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    return model

def create_model1():
    #this network implements a dropout in the input layer with weight constraint
    model = Sequential()
    #adds a dropout with a rate of 20%, meaning one in five inputs will be randomly excluded with each update cycle
    model.add(Dropout(0.2, input_shape=(60,)))
    #adding a "W_constraint" for each hidden layer ensures that the maximum norm of the weights does not exceed a value of 3, which is recommended by the original paper
    model.add(Dense(60, init = 'normal', activation = 'relu', W_constraint=maxnorm(3)))
    model.add(Dense(30, init = 'normal', activation = 'relu', W_constraint=maxnorm(3)))
    model.add(Dense(1, init='normal', activation = 'sigmoid'))
    #compile model
    #also recommended to lift the 'lr' and 'momentum' by one order of magnitude
    #in this case, lr = 0.1 and momentum = 0.9 compared to the baseline model with lr = 0.01 and momentum = 0.8
    sgd = SGD(lr=0.1, momentum = 0.9, decay = 0.0, nesterov = False)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    return model

def create_model2():
    #this network implements dropouts between the two hidden layers, and between the last hidden layer and the output layer
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation = 'relu', W_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, init = 'normal', activation = 'relu', W_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
    sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0.0, nesterov = False)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model2, nb_epoch = 300, batch_size = 16, verbose = 0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y = encoded_Y, n_folds = 10, shuffle = True, random_state = seed)
results = cross_val_score(pipeline, X, encoded_Y, cv = kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
