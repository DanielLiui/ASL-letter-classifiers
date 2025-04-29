# 99% accurate MobileNetV2 and 98% accurate custom CNN for classifying letters of ASL hand sign images
Authors: Lily Czarnecki, Arthur Dao, Daniel Liu

This repository contains code for training a custom keras CNN, a MobileNetV2, and a ResNet50 for classifying letters of ASL hand sign images. The models are also available for download. 

Dataset:
https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/

Model performances for a train-validation-test split of 70-15-15%:

Model       |  Validation accuracy |    Test accuracy
------------|----------------------|----------------------
Custom CNN  |         98%          |        98%
MobileNetV2 |         99%          |        99%
ResNet50    |         81%          |    not obtained

<br>

## Custom Keras CNN 
### Data preprocessing
- 200 x 200 images are read as matrices of pixel values, converted to grayscale, and normalized to a range of [0,1].
- Labels are 1-hot encoded. 

### Architecture
- Convolution layer (filters=32, size=3, ReLu activation); max pooling layer (size 2); drop-out layer (rate=0.2)
- Convolution layer (filters=64, size=3, ReLu activation); max pooling layer (size 2); drop-out layer (rate=0.3)
- Convolution layer (filters=64, size=3), ReLu activation; max pooling layer (size 2); drop-out layer (rate=0.3)
- Dense layer (filters = 128, ReLu activation); drop-out layer (rate = 0.4)
- Output layer (output of 26 classes, softmax activation)

<br>

## MobileNetV2 
### Data preprocessing
- 200 x 200 images are read as matrices of RGB pixel values, resized to 50 x 50, and normalized to a range of [0,1].
- Images are RGB since MobileNetV2 was trained on RGB images. 50 x 50 image size was chosen since it is the optimal image size that is both minimally small for more efficient training and yields high classification accuracy.

### Architecture
- Pretrained weights with a bit of fine-tuning for the last layers

<br>

## How to run
### Installing packages
Can install program packages by activating virtual environment however that is done in in your IDE. 

If you want to install packages on your local device, run:
```
> pip install -r requirements.txt
```

<br>

### Installing dataset
1. Download the ASL Alphabet dataset here: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

2. Move the 'train' folder that contains training images into your project. There is also a 'test' folder that contains test images, but the test set is very small so it is not needed. Each folder contains
folders named 'A', 'B', ... to 'Z' that contains hand sign images of each letter.

3. In preprocess.py, call
```
preprocess_images(dir_name, X_numpy_output_name, y_numpy_output_name, n_images_per_letter, img_size) 
```
to save the input images and output labels as numpy arrays called <X_numpy_output_name>.npy and <y_numpy_output_name>.npy. 
n_images_per_letter is the number of images from the start of each letter folder to use.

<br>
 
### Running the models
- Trained models are already saved at asl_cnn.keras and mobileNetV2.keras. You can load and test them how you like.
- The model training files are keras_CNN.py, mobileNetV2.py, and resnet.py. The main training functions are located near the bottom of the model training files, while the helper files are above.

> [!WARNING]
> You can run the functions how you like, but make sure in preprocess_images() to EDIT THE NAMES OF THE NUMPY FILES saved from preprocess.py.
