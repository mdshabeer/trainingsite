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

