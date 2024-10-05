# CIFAR-10 Image Classification: ANN vs. CNN

This project demonstrates a comparative study of two deep learning models—**Artificial Neural Network (ANN)** and **Convolutional Neural Network (CNN)**—for classifying images from the CIFAR-10 dataset. The aim of this project is to highlight the differences in performance between these two architectures when applied to image classification tasks.

## Dataset

The **CIFAR-10** dataset consists of 60,000 32x32 color images divided into 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is split into 50,000 training images and 10,000 test images.

## Project Overview

This project conducts a **side-by-side comparison** of ANN and CNN models on the same dataset and evaluates their performance using accuracy, confusion matrix, and visualizing misclassified images. The project demonstrates how CNNs, due to their ability to extract spatial features, outperform ANNs on image data.

## Model Architectures

### 1. **Artificial Neural Network (ANN)**

- The ANN model treats the input image as a flattened 1D array and passes it through fully connected dense layers.
- **Architecture:**
  - `Flatten` layer to convert the 32x32x3 image into a 1D vector.
  - Two hidden `Dense` layers with 3000 and 1000 neurons, using ReLU activation.
  - Output `Dense` layer with 10 neurons and softmax activation for multi-class classification.

#### Performance:
- **Training Accuracy:** ~49% (after 5 epochs)
- **Validation Accuracy:** ~49%
  
### 2. **Convolutional Neural Network (CNN)**

- The CNN uses convolutional layers to capture spatial hierarchies in the images, followed by pooling and dense layers.
- **Architecture:**
  - Two `Conv2D` layers with 32 and 64 filters, using ReLU activation.
  - Two `MaxPooling2D` layers to reduce the dimensionality while preserving features.
  - Final `Dense` layer with 64 neurons using ReLU activation, followed by a softmax output layer.

#### Performance:
- **Training Accuracy:** ~70% (after 5 epochs)
- **Validation Accuracy:** ~70%

### Comparison Summary

- **Computation Efficiency:** CNNs are more computationally efficient due to max pooling, which reduces the dimensionality of the image while preserving important features, leading to faster training and better generalization.
- **Accuracy:** CNN significantly outperforms ANN, as it is better at capturing the spatial relationships between pixels, a key aspect of image data.
  
## Conclusion

This project demonstrates the superiority of **Convolutional Neural Networks (CNNs)** over **Artificial Neural Networks (ANNs)** for image classification tasks. While ANNs struggle with high-dimensional data like images, CNNs excel due to their ability to extract spatial features through convolutional operations and pooling layers.   
