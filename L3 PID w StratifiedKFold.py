## using StratifiedKFold for validation

#load libraries
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
import numpy

#fix random seed
seed = 7
numpy.random.seed(seed)

#load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

#split into input (X) and output (Y)
#array[<rows>,<columns>]
#X takes the first 8 columns as input and
#Y takes the last column as output
X = dataset[:,0:8]
Y = dataset[:,8]

# create a 10-fold cross validation test
kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state = seed)
#create an array to store the scores for each fold
cvscores = []

#creates 10 models for each fold
for i, (train, test) in enumerate(kfold):
    #create model
    model = Sequential()
    #use Dense() to create fully connected layers
    model.add(Dense(12, input_dim=8, init='uniform', activation = 'relu'))
    model.add(Dense(8, init = 'uniform', activation = 'relu'))
    model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))

    #compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #fit the model
    #verbose = 0 to turn off verbose output for each epoch
    model.fit(X[train], Y[train], nb_epoch = 150, batch_size = 10, verbose = 0)

    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose = 0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #add the score for each model into cvscores to determine mean score and std dev
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))