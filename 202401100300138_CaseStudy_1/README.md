# Handwritten Digit Recognition (MNIST)

A CNN-based approach to classifying handwritten digits from the MNIST dataset, built with TensorFlow and Keras.

---

## Overview

The goal was to train a model that can look at a 28x28 grayscale image of a handwritten digit and correctly identify which digit (0-9) it represents. The approach goes from raw CSV pixel data all the way to a submission-ready predictions file.

### Preprocessing

Normalized pixel values from the range [0, 255] down to [0, 1] by dividing by 255. Reshaped the flat pixel arrays into 28x28x1 tensors to match the expected input format for a CNN. 

### Model Architecture

Built a Sequential CNN with the following structure:

- Data augmentation layers: random rotation and random translation to improve generalization
- Three Conv2D layers (32, 64, 128 filters) with ReLU activation
- MaxPooling after the first two conv layers to reduce spatial dimensions
- Flatten layer to transition from feature maps to a dense representation
- Dense layer with 128 units and ReLU activation
- Dropout (0.5) to reduce overfitting
- Output Dense layer with 10 units and softmax activation for class probabilities

### Compilation

Compiled the model using the Adam optimizer, sparse categorical crossentropy as the loss function, and accuracy as the evaluation metric.

Trained for 10 epochs with a batch size of 64, using the validation set to monitor performance after each epoch.

### Performance Visualization

Plotted training and validation accuracy and loss curves across epochs to visually assess how well the model learned and whether it was overfitting.

### Evaluation via Confusion Matrix

Generated predictions on the validation set and plotted a 10x10 confusion matrix using seaborn to see which digits the model confused with each other.
