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

seed = 7
numpy.random.seed(seed)

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

#baseline model
def create_baseline():
    #create model
    #we start with same number of neurons as input in hidden layer as a starting point
    model = Sequential()
    model.add(Dense(60, input_dim = 60, init = 'normal', activation = 'relu'))
    model.add(Dense(1, init = 'normal', activation= 'sigmoid'))
    #compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn = create_baseline, nb_epoch = 100, batch_size = 5, verbose = 0)
kfold = StratifiedKFold(y = encoded_Y, n_folds = 10, shuffle = True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv = kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

