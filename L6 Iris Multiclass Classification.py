# import libraries
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix seed
seed = 7
numpy.random.seed(seed)

# load dataset
## it is easier to use 'pandas' to load dataset instead of 'numpy' if output variable contains strings
dataframe = pandas.read_csv("Iris.csv", header = 0)
dataset = dataframe.values
X = dataset[:,1:5].astype(float)
Y = dataset[:,5]

#encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
#encoder.transform() does the conversion into integer
encoded_Y = encoder.transform(Y)
#convert integers to dummy variables (ie one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
## fully connected network with 4 inputs -> [4 hidden nodes] -> 3 outputs
## 3 outputs as we did one hot encoding, so 1 output for each class
def baseline_model():
    model = Sequential()
    model.add(Dense(4, input_dim = 4, init = 'normal', activation = 'relu'))
    ## output uses sigmoid activation function to ensure output values are in the range of 0 and 1
    model.add(Dense(3, init = 'normal', activation = 'sigmoid'))
    #compile model
    ## model uses ADAM gradient descent optimization algo
    ## and uses a logarithmic loss function, called categorical_crossentropy in keras
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch = 100, batch_size = 5, verbose = 0)

#using k-fold validation
kfold = KFold(n=len(X), n_folds=10, shuffle= True, random_state = seed)

results = cross_val_score(estimator, X, dummy_y, cv = kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
