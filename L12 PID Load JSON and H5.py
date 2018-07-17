from keras.models import model_from_json
import numpy

#load model structure and weights
json_file = open("model.json", "r")
loaded_model = model_from_json(json_file.read())
json_file.close()
loaded_model.load_weights("model.h5")
print("Loaded!")

#load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

#need to compile after loading the model, but dont have to fit it again!
loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
score = loaded_model.evaluate(X, Y, verbose = 0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

