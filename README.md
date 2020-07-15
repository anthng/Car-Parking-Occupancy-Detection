# Parking Lot Detection

The implementation of [Real-time image-based parking occupancy detection using deep learning](http://ceur-ws.org/Vol-2087/paper5.pdf) paper. In this paper, the authors perform a pre-trained of Convolutional Neural Networks (VGGNet) and Support Vector Machine in order to detect parking occupancy.

# Dataset: Parking Lot Database

This database contains 12,417 images (1280X720) captured from two different parking lots (parking1 and parking2) in sunny, cloudy and rainy days. The first parking lot has two different capture angles (parking1a and parking 1b).

The images are organised into three directories (parking1a, parking1b and parking2). Each directory contains three subdirectories for different weather conditions (cloudy, rainy and sunny). Inside of each subdirectory the images are organised by acquisition date.

More info about the dataset: https://web.inf.ufpr.br/vri/databases/parking-lot-database/

# Code

* [make_dataset.py](./make_dataset.py) - Build dataset

* [model](./model) - Contain my baseline model and pre-trained model
  * [baseline.py](./model/baseline.py) - My baseline
  * [vgg.py](./model/vgg.py) - Apply the pre-trained VGG16
* [train.py](./train.py) - Run training for model folder
* [extract_feats.py](./extract_feats.py) - Use CNN model to extract features
* [svm_clf_from_cnn_feats.py](./svm_clf_from_cnn_feats.py) - Classifier

# My Result

I can not apply the pre-trained VGG16 model to extract features because of my hardware limitations. I used a MobileNet (is a ight-weight model) instead of VGG16 as mentioned by the authors. The result was not gained as my expectation. 

## My Baseline

![baseline](/imgs/baseline.png)
![baseline](/imgs/baseline_result.png)

## Pre-trained model

![Pre-trained](/imgs/transfer.png)
![Pre-trained](/imgs/vgg_result.png)

## CNN - Feature Extraction and SVM Classifier

![cnn-feats-svm-clf](/imgs/svm.png)
