#import libraries
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#load data
dataframe = pandas.read_csv("sonar.csv", header = None)
dataset = dataframe.values
# split into input (X) and output (Y)
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# one hot encoding
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def create_smaller():
    #create a network with less neurons in first hidden layer
    # 60i - 30 - 1o
    model = Sequential()
    model.add(Dense(30, input_dim = 60, init = 'normal', activation = 'relu'))
    model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
    #compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def create_larger():
    #create a network with more hidden layers
    # 60i - 60 - 30 - 1o
    model = Sequential()
    model.add(Dense(60, input_dim = 60, init = 'normal', activation = 'relu'))
    model.add(Dense(30, init = 'normal', activation = 'relu'))
    model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

seed = 7
numpy.random.seed(seed)

#evaluating the smaller network
estimatorsSmall = []
estimatorsSmall.append(('standardize', StandardScaler()))
estimatorsSmall.append(('mlp', KerasClassifier(build_fn = create_smaller, nb_epoch = 100, batch_size = 5, verbose = 0)))
pipelineSmall = Pipeline(estimatorsSmall)
kfold = StratifiedKFold(y = encoded_Y, n_folds = 10, shuffle = True, random_state = seed)
resultsSmall = cross_val_score(pipelineSmall, X, encoded_Y, cv = kfold)

#evaluating the larger network
estimatorsLarge = []
estimatorsLarge.append(('standardize', StandardScaler()))
estimatorsLarge.append(('mlp', KerasClassifier(build_fn = create_larger, nb_epoch = 100, batch_size = 5, verbose = 0)))
pipelineLarge = Pipeline(estimatorsLarge)
kfold = StratifiedKFold(y = encoded_Y, n_folds = 10, shuffle = True, random_state = seed)
resultsLarge = cross_val_score(pipelineLarge, X, encoded_Y, cv = kfold)

# results
print("Smaller: %.2f%% (%.2f%%)" % (resultsSmall.mean() * 100, resultsSmall.std() * 100))
print("Larger: %.2f%% (%.2f%%)" % (resultsLarge.mean() * 100, resultsLarge.std() * 100))


