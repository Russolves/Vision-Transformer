# Vision-Transformer for Image Classification
This repository contains the implementation of a Vision Transformer (ViT) model for image classification on a custom dataset. The model leverages the power of the transformer architecture to efficiently learn the essential features for classifying images into 5 different categories: airplane, bus, cat, dog, and pizza.

# Features
- Utilizes the COCO dataset to create a custom dataset containing 5 classes (airplane, bus, cat, dog, and pizza), with 1500 images per class
- Employs data augmentation techniques such as random affine transformations, color jitter, and random horizontal flips to improve the model's generalization capabilities
- Implements a Vision Transformer architecture that includes:
  - Conv2D layer for creating image patch embeddings
  - Class token and position embeddings
  - Multi-head self-attention and Transformer encoder
  - MLP layer for final class prediction
- Trains the model using the Adam optimizer and CrossEntropyLoss for 15 epochs
- Evaluates the model's performance using a confusion matrix
- Saves the trained model and plots the training loss graph

# Usage
1. Ensure that the COCO dataset is available in the specified path (can alter path based on personal directory structure)
2. Run the main script to create the custom dataset, train the ViT model, and evaluate its performance

*Note that in order to significantly reduce computational complexity, it is recommended to run the image resizing function first before main which would turn all the pyCOCO dataset images into 64x64 images

# Dependencies
- Python 3.6+
- Pytorch
- torchvision
- pycocotools
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- PIL (Python Imaging Library)

# Reference
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929.
- Avinash Kak's ViTHelper.py (also included within repository)
