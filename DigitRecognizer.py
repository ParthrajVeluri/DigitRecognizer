import cv2 as cv
import ctypes
from PIL import Image
import tensorflow as tf
import numpy as np

win = np.ones((400,400,3), dtype = 'float64')

run = False

m = "FFW"

#Loading Model 
if m == "FFWD":
    model = tf.keras.models.load_model("DigitRecognizer/trained_model")
    model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['acc'])
else:
    model = tf.keras.models.load_model("DigitRecognizer/CNNtrained_model")
    model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['acc'])

def draw(event, x, y, flag, param):
    global run

    #Start Drawing on click
    if event == cv.EVENT_LBUTTONDOWN:
        run = True
    
    #Stop Drawing on release
    if event == cv.EVENT_LBUTTONUP:
        run = False

    #Keep Drawing on hold
    if event == cv.EVENT_MOUSEMOVE: 
        if run == True:
            cv.rectangle(win, (x,y), (x+30, y+30) ,(0,0,0), -1)


cv.namedWindow('window')
cv.setMouseCallback('window', draw) #On any mouse action, will execute draw func

while True: 
    cv.imshow('window', win)

    k = cv.waitKey(1) #Listen for inputs

    if k == 27: #If esc is pressed
        cv.destroyAllWindows()
        break
    
    if k == 32: #If space is pressed
        #Saving Image (To learn how to process the image)
        im = Image.fromarray((win * 255).astype(np.uint8))
        im.save("DigitRecognizer/img.jpg")

        #Preprocessing the loaded image to send into the NeuralNet
        #Read the image
        loadedImage = cv.imread('DigitRecognizer/img.jpg')
        #Convert Image to grayscale
        loadedImage = cv.cvtColor(loadedImage, cv.COLOR_BGR2GRAY)
        #Threshold 
        threshold, loadedImage = cv.threshold(loadedImage,10.5,255, cv.THRESH_BINARY_INV)
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

        #make shape 1,28,28 as model needs a None, 28, 28 input shape
        loadedImage = np.expand_dims(loadedImage, axis = 0)

        #Note 1-loaded because model was trained with 
        #black bg and white text while I am doing 
        #White bg and black text
        pred = model.predict(loadedImage)
        prediction = np.argmax(pred[0])
        print(prediction, pred[0][prediction])
        ctypes.windll.user32.MessageBoxW(0, f"Prediction: {prediction}\n Confidence: {pred[0][prediction]}", "Prediction", 1)
        win = np.ones((400,400,3), dtype = 'float64')

