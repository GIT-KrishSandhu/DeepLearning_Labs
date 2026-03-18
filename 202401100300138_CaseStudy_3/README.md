# Brain Tumor Detection

A binary image classification CNN to detect the presence of a brain tumor in grayscale MRI scans, built with TensorFlow and Keras.

---

## Overview

The goal was to train a model that looks at an MRI scan and predicts one of two outcomes: tumor present (yes) or not present (no). The dataset is organized into two labeled folders and the pipeline goes from file path collection to a saved production-ready model.

### Exploratory Check

Sampled 10 random rows from the DataFrame and printed class distribution to confirm the dataset was reasonably balanced between tumor and non-tumor cases.

### Image Data Generator

Used Keras `ImageDataGenerator` with `flow_from_dataframe` to stream images from disk for both train and test sets. Images were resized to 224x224, loaded in grayscale, and batched in groups of 16. Binary class mode was used to match the two-class setup.

### Model Architecture

Built a Sequential CNN for binary classification:

- Three Conv2D blocks (32, 64, 128 filters) with ReLU activation, each followed by MaxPooling
- Dropout (0.25) after the first two conv blocks to reduce overfitting
- Flatten layer to transition to dense layers
- Three Dense layers (128, 64, 32 units) with ReLU activation
- Dropout (0.5) before the dense stack
- Output Dense layer with 1 unit and sigmoid activation for binary probability

### Compilation

Compiled with the Adam optimizer, binary crossentropy loss, and accuracy as the metric.

Trained for 35 epochs using the test generator as validation data to track generalization across each epoch.

### Visualization

Plotted accuracy and loss curves for both train and validation sets across all 35 epochs to assess learning stability and overfitting.

Randomly sampled 3 images from each class folder and ran inference on each. Displayed the MRI alongside its actual label and predicted result, color-coded red for Tumor and green for No Tumor.