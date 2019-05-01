# CNN Image Classifier

## Instructions:
 - This program was compiled and ran in python 3.7.1 inside the Anaconda environment containing TensorFlow backend and Keras library for using the CNN.
 - The libraries used inside the program besides Keras were os, sys, numpy, and PIL.
 - The file image_augmentation.py increases the amount of non-corroded images as there was a minimal amount in comparison to corroded images in the initial file.
 - train-binary.py trains the CNN on the training dataset, which must be run first in order to create values that can be used for the validation of the images.
 - predict-binary.py uses the model information gained from running train-binary.py and tests the validity of the CNN on a test dataset.
 
## Image_Augmentation.py:
 - Increases the amount of non-corroded images.  Need only be ran once.
 
## Train-Binary.py:
 - Train's the CNN, images saved to a h5 file.  Needs to be ran before predict-binary.py.
 
## Predict-Binary.py:
 - Predicts the test images based on the CNN information saved within the h5 file from train-binary.py.  Must be ran after train-binary.py.
