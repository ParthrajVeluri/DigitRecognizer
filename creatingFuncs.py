import tensorflow as tf
import numpy as np
from matplotlib import image, pyplot
import cv2 as cv

model = tf.keras.models.load_model("DigitRecognizer/trained_model")
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['acc'])

#Preprocessing the loaded image to send into the NeuralNet
#Read the image
loadedImage = cv.imread('DigitRecognizer/img.jpg')
#Convert Image to grayscale
loadedImage = cv.cvtColor(loadedImage, cv.COLOR_BGR2GRAY)
#Threshold 
threshold, loadedImage = cv.threshold(loadedImage,10.5,255, cv.THRESH_BINARY_INV)
print(loadedImage, threshold)

#Mapping pixels from 0 to 1 (0 is black)
loadedImage = loadedImage/255
#Making shape 28,28,1 because resize method needs 3 dimensions
loadedImage = np.expand_dims(loadedImage, axis = 2)
#Resizes the image into a 28x28
loadedImage = tf.image.resize(loadedImage, (28, 28), antialias = False)
#Get rid of the extra dimension added before to get shape 28,28
loadedImage = loadedImage[:, :, 0]
#previous line returns a tensor, convert tensor to numpy array
loadedImage = loadedImage.numpy()

pyplot.imshow(loadedImage, cmap = "gray")

#make shape 1,28,28 as model needs a None, 28, 28 input shape
loadedImage = np.expand_dims(loadedImage, axis = 0)
print(loadedImage.shape)

#Note 1-loaded because model was trained with 
#black bg and white text while I am doing 
#White bg and black text
pred = model.predict(loadedImage)
prediction = np.argmax(pred[0])
print(prediction)
print(pred[0][prediction])

pyplot.show()