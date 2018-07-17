## Regression problems are those whose outputs are a range of values

#import libraries
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
# note that dataset is whitespace separated
dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header = None)
dataset = dataframe.values
# split into input(X) and output(Y)
X = dataset[:,0:13]
Y = dataset[:,13]

#base model
def baseline_model():
    #create model
    model = Sequential()
    # 13i - 13 - 1o
    model.add(Dense(13, input_dim = 13, init = 'normal', activation = 'relu'))
    # no activation function is used for output layer because this is a regression problem
    #and we are interested in predicting numerical values directly without transform
    model.add(Dense(1, init = 'normal'))
    model.compile(loss = 'mean_squared_error', optimizer='adam')
    return model

def larger_model():
    # increase layers
    #create model
    model = Sequential()
    # 13i - 13 - 6 - 1o
    model.add(Dense(13, input_dim = 13, init = 'normal', activation = 'relu'))
    model.add(Dense(6, init='normal', activation='relu'))
    # no activation function is used for output layer because this is a regression problem
    #and we are interested in predicting numerical values directly without transform
    model.add(Dense(1, init = 'normal'))
    model.compile(loss = 'mean_squared_error', optimizer='adam')
    return model

def wider_model():
    # shallow network but with more neurons in the one hidden layer
    #create model
    model = Sequential()
    # 13i - 20 - 1o
    model.add(Dense(20, input_dim = 13, init = 'normal', activation = 'relu'))
    # no activation function is used for output layer because this is a regression problem
    #and we are interested in predicting numerical values directly without transform
    model.add(Dense(1, init = 'normal'))
    model.compile(loss = 'mean_squared_error', optimizer='adam')
    return model

seed = 7
numpy.random.seed(seed)

estimators = []
#standardize the data
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=100,batch_size=5, verbose = 0)))
pipeline = Pipeline(estimators)
#evaluating the model
kfold = KFold(n = len(X), n_folds=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)

estimatorsLarger = []
#standardize the data
estimatorsLarger.append(('standardize', StandardScaler()))
estimatorsLarger.append(('mlp', KerasRegressor(build_fn=larger_model, nb_epoch=100,batch_size=5, verbose = 0)))
pipelineLarger = Pipeline(estimatorsLarger)
#evaluating the model
kfold = KFold(n = len(X), n_folds=10, random_state=seed)
resultsLarger = cross_val_score(pipelineLarger, X, Y, cv=kfold)

estimatorsWider = []
#standardize the data
estimatorsWider.append(('standardize', StandardScaler()))
estimatorsWider.append(('mlp', KerasRegressor(build_fn=wider_model, nb_epoch=100,batch_size=5, verbose = 0)))
pipelineWider = Pipeline(estimatorsWider)
#evaluating the model
kfold = KFold(n = len(X), n_folds=10, random_state=seed)
resultsWider = cross_val_score(pipelineWider, X, Y, cv=kfold)

print("Results: %.2f (%.2f) MSE" % (abs(results.mean()), results.std()))
print("Larger: %.2f (%.2f) MSE" % (abs(resultsLarger.mean()), resultsLarger.std()))
print("Wider: %.2f (%.2f) MSE" % (abs(resultsWider.mean()), resultsWider.std()))

################################
# Results: 567.31 (277.51) MSE #
# Larger: 576.33 (288.83) MSE  #
# Wider: 560.62 (272.04) MSE   #
################################