# Machine-Learning-Python
### Journey towards Machine Learning Mastery!
---

This repository contains the various deep learning models that were developed while learning Machine Learning.
The models created are based on Tensorflow backend and uses Keras.

Lesson 1: Simple calculation using basic Tensorflow functions

Lesson 2: Creating a model on Pima Indian Diabetes dataset, with a validation split of 0.33

Lesson 3: Creating a model on Pima Indian Diabetes dataset, using 'StratifiedKFold' class to perform K-fold cross validation

Lesson 4: Using 'KerasClassifier' wrapper to create model on Pima Indian Diabetes dataset

Lesson 5: Performing Grid Search to evaluate different configurations for the neural network. Model works on Pima Indian Diabetes dataset

Lesson 6: Creating a fully connected network for the famous Iris flower multiclass classification problem. One hot encoding is performed to encode class values as integers

Lesson 7: A fully connected network is created for the Sonar Object Classification problem

Lesson 8: Perform standardization on the data before building the network. A pipeline is created to standardize the data in each pass of the cross validation instead of standardizing the entire dataset

Lesson 9: Tuning the network by experimenting with different network structures. A smaller network with fewer neurons in the first hidden layer, as well as a larger network with an additional hidden layer are created and tested.

Lesson 10: Compared three neural network structures on Boston House Prices Regression problem - the baseline model, a larger model with more hidden layers, and a wider model with more neurons in the hidden layer

Lesson 11: Save model into JSON and weights into HDF5 format for later use. Model is trained on Pima Indian Diabetes dataset

Lesson 12: Model is loaded from JSON and weights are loaded from HDF5 file.

Lesson 13: Checkpointing neural network improvements over epochs. 2 checkpointing functions are created here. The first function creates a hdf5 file and saves the new weights every time there is an improvement in accuracy. The second function creates 1 hdf5 file and updates the file everytime there is an update, hence the hdf5 file contains the weights with the best accuracy at the end of the training. Model is trained on Pima Indian Diabetes dataset.

Lesson 14: Loading HDF5 containing the weights with best accuracy. Model is trained on Pima Indian Diabetes dataset

Lesson 15: Uses pyplot to plot the history of training and test accuracy and loss.  Useful way to estimate whether we should increase or decrease number of epochs to improve performance

Lesson 16: Implemented dropout to improve performance. Model is trained on Sonar dataset.

Lesson 17: Implemented time-based learning rate schedule on Ionosphere dataset

Lesson 18: Implemented drop-based learning rate schedule on Ionosphere dataset

Lesson 19: Plot 9 of the image inputs in a 3x3 grid on MNIST dataset

Lesson 20: Trained a network to recognise MNIST data using Multi-layered Perceptrons

Lesson 21: Trained a network to recognise MNIST data using Convolutional Neural Networks (CNNs)

Lesson 22: Performed object recognition using CNNs on CIFAR10 dataset

Lesson 23: Performed sentiment analysis using CNNs on IMDB dataset
