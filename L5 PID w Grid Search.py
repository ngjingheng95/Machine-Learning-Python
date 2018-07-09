## Testing different configurations of optimizer, init, epoch and batchsize
## using GridSearchCV()

# load libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import numpy
import pandas

# define create_model function for KerasClassifier wrapper
# include default values for grid search, in this case, optimizer = 'rmsprop', init = 'glorot_uniform'
def create_model(optimizer = 'rmsprop', init = 'glorot_uniform'):
    # create model
    model = Sequential()
    # 8 -> 12 -> 8 -> 1
    model.add(Dense(12, input_dim=8, init = init, activation = 'relu'))
    model.add(Dense(8, init = init, activation = 'relu'))
    model.add(Dense(1, init = init, activation = 'sigmoid'))
    #compile model
    model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    return model

# fix seed
seed = 7
numpy.random.seed(seed)

# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:,8]

# create model using KerasClassifier class
model = KerasClassifier(build_fn=create_model)

# create arrays of optimizer, init, epoch and batchsize for grid search
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = numpy.array([50,100,150])
batches = numpy.array([5,10,20])

# create a dict for grid search
param_grid = dict(optimizer = optimizers, nb_epoch = epochs, batch_size = batches, init = init)
grid = GridSearchCV(estimator = model, param_grid=param_grid)
grid_result = grid.fit(X, Y)

# summarize results
#gives the best result
print("best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#gives all the results
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))