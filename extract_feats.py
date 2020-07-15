import os
from glob import glob

import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Input, BatchNormalization
 
from model.baseline import MyModel
import numpy as np
import cv2

from tensorflow.keras.optimizers import SGD


TRAIN_DIR = './dataset/PKLot/custom_dataset/train/'
VALID_DIR = './dataset/PKLot/custom_dataset/valid/'
ROOT_DIR = '../../dataset/Pklot/PKLotSegmented/'

WIDTH, HEIGHT = 39,72

NB_EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 32
# print("Setup model")

#base_model  = load_model('./output')
base_model = MobileNetV2(weights='imagenet', include_top=True)
print(base_model.summary())

model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
#opt = SGD(lr = LR, momentum=0.9, decay = LR/NB_EPOCHS)

#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


def extract_features(list_img_path, features_path = './features',label_path_to_save = './'):
      ground_truths = []
      for img in list_img_path:
            img = img.replace("\\","/")
            label = img.split("/")[-2]
            img_name = img.split("/")[-1]
            img_name = img_name.replace(".jpg", ".npy")
            if label == "Empty":
                  ground_truths.append(0)
            else:
                  ground_truths.append(1)

            image = cv2.imread(img)
            image = cv2.resize(image, (224, 224))
            image_x = np.expand_dims(image, axis=0)
            image_x = preprocess_input(image_x)
            feature = model.predict(image_x)
            os.makedirs(os.path.dirname(features_path), exist_ok=True)
            np.save(features_path + img_name, feature)

      np.save(label_path_to_save,ground_truths)

if __name__ == "__main__":
      # occupied_val = VALID_DIR + 'Occupied/*.jpg'
      # empty_val = VALID_DIR + 'Empty/*.jpg'
      # valid_images = list(glob(occupied_val) + glob(empty_val))

      occupied_train = TRAIN_DIR + 'Occupied/*.jpg'
      empty_train = TRAIN_DIR + 'Empty/*.jpg'
      train_images = list(glob(occupied_train) + glob(empty_train))

      extract_features(train_images, './features/PKLot/Train/', './features/PKLot/train_label.npy')
      
      occupied_valid = VALID_DIR + 'Occupied/*.jpg'
      empty_valid = VALID_DIR + 'Empty/*.jpg'
      valid_images = list(glob(occupied_valid) + glob(empty_valid))
      extract_features(valid_images, './features/PKLot/Valid/', './features/PKLot/valid_label.npy')