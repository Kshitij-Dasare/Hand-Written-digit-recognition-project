# Handwritten Digit Recognition using PyGame and CNN

## 📌 Project Overview
The Handwritten Digit Recognition project utilizes PyGame to create an interactive drawing canvas where users can sketch digits (0-9), which are then recognized using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model achieves an accuracy of ~99% on MNIST. The PyGame interface includes a drawing area, a clear button, and real-time prediction display. The project leverages Python, PyGame, TensorFlow/Keras, OpenCV, and NumPy for efficient image processing and accurate classification.

---

## 📂 Dataset: MNIST Handwritten Digits
### 🔹 About the MNIST Dataset
The MNIST dataset (Modified National Institute of Standards and Technology) is a benchmark dataset for handwritten digit recognition. It consists of:
- 60,000 training images (grayscale, 28x28 pixels)
- 10,000 test images (grayscale, 28x28 pixels)
- Digit labels from 0 to 9

### 🔹 Preprocessing Steps
The dataset is preprocessed using NumPy and Keras utilities:
1. Convert the images to grayscale.
2. Normalize pixel values to range [0,1] by dividing by 255.
3. Reshape images to (28,28,1) to match CNN input requirements.
4. Convert labels to one-hot encoding using `keras.utils.to_categorical()`.

```python
import numpy as np
import keras
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
```

---

## 🎮 PyGame Module
### 🔹 About PyGame
[PyGame](https://www.pygame.org/) is a cross-platform library used for game development and interactive applications. In this project, PyGame is used to:
- Create a drawing canvas where users can write digits using the mouse.
- Capture the drawn digit and process it for model prediction.
- Display the predicted digit on the screen.

### 🔹 PyGame Implementation Overview
The PyGame script initializes a drawing canvas where users can write digits using the mouse. The drawn digit is captured and preprocessed before being fed into the trained CNN model. The model predicts the digit and displays the output on the screen.

```python
import pygame, sys
import numpy as np
import cv2
from keras.models import load_model

pygame.init()
DISPLAYSURF = pygame.display.set_mode((640, 480))
MODEL = load_model("bestmodel.h5")

# Main loop to capture user input, preprocess image, and predict digit
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
```

---

## 🧠 CNN Model for Digit Recognition
### 🔹 Model Architecture
The CNN model consists of:
- Conv2D (32 filters, 3×3 kernel, ReLU activation, input shape (28,28,1))
- MaxPooling2D (2×2 pool size)
- Conv2D (64 filters, 3×3 kernel, ReLU activation)
- MaxPooling2D (2×2 pool size)
- Flatten layer to convert 2D data into a 1D vector
- Dropout (0.5) to prevent overfitting
- Dense layer with 10 neurons and softmax activation for classification

```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```

---

## 🚀 Running the Project
Run the following command to start the application:
```bash
python digit_recognition.py
```

---

## 🤝 Contributing
Feel free to fork this repository, open issues, and submit pull requests!

---

