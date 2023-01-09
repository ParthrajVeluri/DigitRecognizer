# DigitRecognizer
Application to Recognize Handwritten Digits

This python application was created mainly using tensorflow and keras libraries and opencv for the GUI and image processing. The model was trained using a feed forward neural network and a convolutional neural network on the MNIST digits dataset. Both models perform relatively the same and it is hard to differentiate on their performance. The trained models were evaluated to have 98%+ accuracy. However, it was evaluated using MNIST test sets. In reality, the accuracy seems to be worse mainly due to inputs which differ from the images the model was trained on. When the user draws a number covering the entire canvas, the model will guess with a high accuracy and a high confidence. In other cases outlined below, this is not the case. 

Images taken using the CNN model.

User Draws a number covering the entire canvas. This is good behavior.

<img width="411" alt="Screenshot_20230109_032628" src="https://user-images.githubusercontent.com/58951561/211402859-1d172bd0-8da8-418f-a4b2-4b6810ccb68c.png">

Model guesses 2 but it is not confident. This is because 2 and 7 share some similarity in the way they are drawn. This can be considered somewhat good behavior.

<img width="304" alt="Screenshot_20230109_032715" src="https://user-images.githubusercontent.com/58951561/211403000-cbf99a13-5df8-4cd4-b36c-3adf1f4eb8e0.png">

User gives an invalid input, thus model is guessing here. The input is invalid because the model was only trained on centered images covering the entire canvas. This is good behavior. 

<img width="302" alt="Screenshot_20230109_032823" src="https://user-images.githubusercontent.com/58951561/211403023-a0d4ccb7-9777-4b56-bdfa-59ad79a2dd97.png">

User gives an invalid input but the model is confident it is a 8. This is bad behavior. 

<img width="300" alt="Screenshot_20230109_032853" src="https://user-images.githubusercontent.com/58951561/211403055-3928441b-2dbb-4fcd-afbe-7baa5334a432.png">

