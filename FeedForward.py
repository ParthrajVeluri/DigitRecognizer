import tensorflow as tf
import numpy as np
import keras

#Importing Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path = "mnist.npz")

#Dataset shapes 
print("X Train:",x_train.shape,"Y Train:",y_train.shape,"X Test:",x_test.shape,"Y Test:",y_test.shape)

#Rescaling the data so each pixel ranges from 0 to 1
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
print(x_test.shape)

#Number Labels
labels = [0,1,2,3,4,5,6,7,8,9]

#Creating Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = 'relu'), 
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['acc'])

#Training The model
history = model.fit(x_train, y_train, epochs = 5)
model.save('DigitRecognizer/trained_model', include_optimizer = True, save_format = "h5")

model = tf.keras.models.load_model("DigitRecognizer/trained_model")
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['acc'])

model.summary()

loss, accuracy = model.evaluate(x_test, y_test)
pred = model.predict(x_test)


#for i in range(len(x_test)):
#    print("Actual", y_test[i], "Prediction", np.argmax(pred[i]))

