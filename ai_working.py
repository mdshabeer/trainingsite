1.
This code is used to create, train, and evaluate a neural network model for handwriting recognition, specifically for classifying digits (0-9) using the MNIST dataset, and it includes examples of making predictions on new data.
- Code imports libraries including NumPy, Matplotlib, and TensorFlow Keras; loads and preprocesses MNIST data.
- Normalizes image data to [0,1], one-hot encodes labels, and constructs a Sequential model with Flatten and Dense layers.
- Compiles and trains the model on MNIST data for digit classification, validates on test data.
- After training, code predicts output for a random image and a specific pre-saved image while handling potential exceptions.
  
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the images to be values in range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 images
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 classes for digits 0-9

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Generate a simple random image in the expected format
simple_random_img = np.random.rand(1, 28, 28, 1).astype(np.float32)

# Try to make a prediction with the model
try:
    simple_pred = model.predict(simple_random_img)
    print("Simple test prediction result:", simple_pred)
except Exception as e:
    print("An error occurred during a simple prediction:", e)

from tensorflow.keras.preprocessing import image
import numpy as np

# Assuming model is already defined, compiled and trained;

# Load an image saved earlier
test_image = image.load_img("mnist_test_image_0.png", color_mode='grayscale', target_size=(28, 28))

# Prepare the image as a tensor
test_image_array = np.array(test_image).reshape(1, 28, 28, 1).astype(np.float32) / 255.0

# Invert colors to match MNIST data if necessary (white digit on a black background)
test_image_array = 1 - test_image_array

# Attempt a prediction
try:
    test_pred = model.predict(test_image_array)
    predicted_class = np.argmax(test_pred, axis=-1)
    print("Predicted class for 'mnist_test_image_0.png':", predicted_class)
except Exception as e:
    print("An error occurred during the test image prediction:", e)

2.a. 
The code builds a Convolutional Neural Network (CNN) using Keras to classify handwritten digits from the MNIST dataset, implementing early stopping during training and saving the model.

- Preprocesses MNIST images by normalizing pixels and one-hot encoding digit labels.
- Constructs a CNN with two convolutional layers, batch normalization, max pooling, dropout for regularization, and dense layers for classification.
- Compiles the model with the Adam optimizer and categorical crossentropy loss, and trains using early stopping to prevent overfitting.
- Saves the trained model to a file named 'advanced_mnist_model.h5' for later use or inference.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the images to be values in range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Make sure you add the `early_stopping` callback to the `fit()` method
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the model
model.save('advanced_mnist_model.h5')

  
2.b.
The code loads a pre-trained Keras neural network model to predict the class of a digit from a saved grayscale image of size 28x28 pixels.

- Loads a saved Keras model specialized in recognizing MNIST digits.
- Implements a function to load an image, convert it to grayscale, resize to 28x28 pixels, and invert colors if necessary.
- Adds another function to predict the digit by leveraging the previously loaded neural network model.
- Executes the image loading and prediction functions on a sample image to output the classified digit result.
  
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model('advanced_mnist_model.h5')

def load_and_prepare_image(image_path):
    # Load an image saved earlier
    test_image = image.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    # Prepare the image as a tensor
    test_image_array = np.array(test_image).reshape(1, 28, 28, 1).astype(np.float32) / 255.0
    # Invert colors if necessary
    test_image_array = 1 - test_image_array
    return test_image_array
    
def make_prediction(model, prepared_image):
    # Attempt a prediction
    test_pred = model.predict(prepared_image)
    predicted_class = np.argmax(test_pred, axis=-1)
    return predicted_class

# Use the functions
image_path = "mnist_test_image_0.png"
prepared_image = load_and_prepare_image(image_path)
predicted_class = make_prediction(model, prepared_image)
print("Predicted class for 'mnist_test_image_0.png':", predicted_class)
