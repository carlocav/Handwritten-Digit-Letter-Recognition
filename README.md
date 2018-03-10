In this project I built a simple Neural Network with Tensorflow aimed at classifying handwritten digits and characters.

The Neural Network was trained on the EMNIST (Extended MNIST) Balanced Dataset and is composed of 3 convolutional layers, 2 max pooling layers and 2 dense layers with dropout regularization.


In order to test it I also created a Shiny app (using [this repository](https://github.com/rocalabern/HandwrittenDigitRecognition) as a starting point), which required the import into R of the trained neural network ready to be used for the recognition of alphanumerical characters.

In the web app I implemented a conversion process similar to the one used to get the EMNIST dataset which basically means the removal of the white spaces from the starting image, the application of a Gaussian filter, and the resizing to 28x28.

The Shiny app is available [here](http://carlocavalieri.com/recognizer).
