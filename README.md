# Parking Lot Detection

The implementation of [Real-time image-based parking occupancy detection using deep learning](http://ceur-ws.org/Vol-2087/paper5.pdf) paper. In this paper, the auothors perform a pre-trained of Convolutional Neural Networks (VGGNet) and Support Vector Machine in order to detect parking occupancy.

# Dataset: Parking Lot Database

This database contains 12,417 images (1280X720) captured from two different parking lots (parking1 and parking2) in sunny, cloudy and rainy days. The first parking lot has two different capture angles (parking1a and parking 1b).

The images are organised into three directories (parking1a, parking1b and parking2). Each directory contains three subdirectories for different weather conditions (cloudy, rainy and sunny). Inside of each subdirectory the images are organised by acquisition date.

More info about the dataset: https://web.inf.ufpr.br/vri/databases/parking-lot-database/

# Code

* [make_dataset.py](./make_dataset.py) - Build dataset

* [model.py](./model.py) - My baseline model using CNN

* [cnn_svm.py](./) - CNN extracts the features and a binary SVM classifier to detect the occupancy of parking spaces

* [train.py](./train.py) - Run training phase
