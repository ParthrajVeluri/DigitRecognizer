import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

#Importing Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path = "mnist.npz")

#Dataset shapes 
print("X Train:",x_train.shape,"Y Train:",y_train.shape,"X Test:",x_test.shape,"Y Test:",y_test.shape)

#Rescaling the data so each pixel ranges from 0 to 1
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
print(x_test.shape)

#Creating Model
model = keras.models.Sequential()
#Note: 28,28,1 shows its a 2d img in grayscale. 28,28,3 would mean RGB image
#Conv2D(#of filters, dimension of each filter)
model.add(keras.layers.Conv2D(32, (6,6), activation = 'relu', padding = 'same', input_shape = (28,28,1)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64, (6,6), activation = 'relu', padding = 'same'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(64, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.summary()

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['acc'])

#Training The model
history = model.fit(x_train, y_train, epochs = 3)
model.save('DigitRecognizer/CNNtrained_model', include_optimizer = True, save_format = "h5")

#model = tf.keras.models.load_model("DigitRecognizer/CNNtrained_model")
#model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['acc'])

#model.summary()

loss, accuracy = model.evaluate(x_test, y_test)
pred = model.predict(x_test)

#for i in range(len(x_test)):
#    print("Actual", y_test[i], "Prediction", np.argmax(pred[i]))

