from tkinter import *
from PIL import Image, ImageDraw
import PIL
import tensorflow as tf
import numpy as np
from matplotlib import image

app = Tk()

app.geometry("800x500")

canvas = Canvas(app)
canvas.pack(expand=1, anchor = 'nw', fill = 'both')

#Drawing Window
frame = Frame(app)
frame.place(relx = 0.05, rely = 0.05, relwidth = 0.6, relheight = 0.8)

canv_frame = Canvas(frame, bg = "white")
canv_frame.pack(expand=1, anchor = 'nw', fill = 'both')

canv_frame.update()
app.update_idletasks()
img = Image.new("1", (canv_frame.winfo_width(), canv_frame.winfo_height()), color = "white")
draw = ImageDraw.Draw(img)

#Get last x and y cursor position
def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

#Draw Line 
def draw_line(event): 
    global lasx, lasy
    canv_frame.create_line((lasx, lasy, event.x, event.y), fill = "black", width = 50)
    draw.line((lasx, lasy, event.x, event.y), fill = "black", width = 20)
    lasx, lasy = event.x, event.y

def export_image():
    #canv_frame.to_file("img.jpg
    img.save("DigitRecognizer/img.jpg")

canv_frame.bind("<Button-1>", get_x_and_y)
canv_frame.bind("<B1-Motion>", draw_line)

#Prediction 
pred_lbl = Label(canvas, text = "Prediction: ", font = 25)
pred_lbl.place(relx = 0.7, rely = 0.2)

#Accuracy
acc_lbl = Label(canvas, text = "Accuracy: ", font = 25)
acc_lbl.place(relx = 0.7, rely = 0.3)

#Go Button 
button = Button(canvas, text = "Predict", padx = 5, pady = 5, command = lambda:[export_image(), make_prediction(pred_lbl, acc_lbl)])
button.place(relx = 0.8, rely = 0.1)

#Importing Model
model = tf.keras.models.load_model("DigitRecognizer/trained_model")
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['acc'])

def make_prediction(predLabel, accLabel):
    #Preprocessing the loaded image to send into the NeuralNet
    #Read the image
    loadedImage = image.imread("DigitRecognizer/img.jpg", format = "jpg")
    #Mapping pixels from 0 to 1 (0 is black)
    loadedImage = loadedImage/255
    #Making shape 28,28,1 because resize method needs 3 dimensions
    loadedImage = np.expand_dims(loadedImage, axis = 2)
    #Resizes the image into a 28x28
    loadedImage = tf.image.resize(loadedImage, (28, 28), antialias = True)
    #Get rid of the extra dimension added before to get shape 28,28
    loadedImage = loadedImage[:, :, 0]
    #previous line returns a tensor, convert tensor to numpy array
    loadedImage = loadedImage.numpy()
    #make shape 1,28,28 as model needs a None, 28, 28 input shape
    loadedImage = np.expand_dims(loadedImage, axis = 0)

    #Note 1-loaded because model was trained with 
    #black bg and white text while I am doing 
    #White bg and black text
    pred = model.predict(1-loadedImage)
    prediction = np.argmax(pred[0])
    predLabel.config(text = f"Prediction: {prediction}")
    accLabel.config(text = f"Prediction: {pred[0][prediction]}")


app.mainloop()


