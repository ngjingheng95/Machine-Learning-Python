## Loading the MNIST Dataset
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load the MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#plot 4 images as grayscale
#plt.subplot(<nrow><ncol><index>)
#for eg plt.subplot(331) plots the image at the top right corner of a 3x3
plt.subplot(331)
plt.imshow(X_train[0], cmap = plt.get_cmap('gray'))
plt.subplot(332)
plt.imshow(X_train[1], cmap = plt.get_cmap('gray'))
plt.subplot(333)
plt.imshow(X_train[2], cmap = plt.get_cmap('gray'))
plt.subplot(334)
plt.imshow(X_train[3], cmap = plt.get_cmap('gray'))
plt.subplot(335)
plt.imshow(X_train[4], cmap = plt.get_cmap('gray'))
plt.subplot(336)
plt.imshow(X_train[5], cmap = plt.get_cmap('gray'))
plt.subplot(337)
plt.imshow(X_train[6], cmap = plt.get_cmap('gray'))
plt.subplot(338)
plt.imshow(X_train[7], cmap = plt.get_cmap('gray'))
plt.subplot(339)
plt.imshow(X_train[8], cmap = plt.get_cmap('gray'))
plt.show()

