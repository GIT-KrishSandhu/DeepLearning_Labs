# Image Classification on CIFAR-10

A CNN-based approach to classifying color images across 10 object categories, built with TensorFlow and Keras.

---

## Overview

The goal was to train a model that can look at a 32x32 RGB image and correctly identify which of 10 classes it belongs to: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck. The dataset is loaded directly through Keras and the pipeline goes from raw images to evaluated predictions.

### Data Loading

Loaded the CIFAR-10 dataset using `keras.datasets.cifar10`, which returns pre-split train and test sets of labeled color images.

### Preprocessing

- Normalized pixel values from [0, 255] to [0, 1] by dividing by 255. No reshaping was needed since the images are already in 32x32x3 format.

- Displayed a 3x3 grid of sample training images with their corresponding class labels to get a feel for what the data looks like across categories.

Defined a sequential augmentation pipeline applied during training:

- Random horizontal flip
- Random rotation (up to 10%)
- Random zoom (up to 10%)

This was done to improve generalization by exposing the model to varied orientations and scales of the same images.

### Model Architecture

Built a Sequential CNN with the following structure:

- Augmentation block as the first layer (applied only during training)
- Three Conv2D layers (32, 64, 64 filters) with ReLU activation
- MaxPooling after the first two conv layers to reduce spatial dimensions
- Flatten layer to transition to dense representation
- Dense layer with 128 units and ReLU activation
- Output Dense layer with 10 units (raw logits, no softmax)

### Compilation

Compiled with the Adam optimizer and sparse categorical crossentropy loss configured with `from_logits=True` to match the raw output of the final layer. Accuracy was tracked as the evaluation metric.

Trained for 10 epochs using the test set as validation data to monitor generalization after each epoch.

### Prediction and Visualization

Ran inference on the test set, used argmax to pick the predicted class, and displayed individual test images alongside their predicted labels to visually verify outputs.
