import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from glob import glob

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2

import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img

from tensorflow.keras.optimizers import SGD

from model.baseline import MyModel
from model.vgg import TransferModel


WIDTH, HEIGHT = 39,72
NB_TRAIN_SAMPLES = 5000
NB_VALID_SAMPLES = 500
CLASSES = ['Empty', 'Occupied'] #empty samples is less than occupied

NB_EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 32


TRAIN_DIR = './dataset/PKLot/custom_dataset/train/'
VALID_DIR = './dataset/PKLot/custom_dataset/valid/'
ROOT_DIR = '../../dataset/Pklot/PKLotSegmented/'

occupied_dir = TRAIN_DIR + 'Occupied/*.jpg'
empty_dir = TRAIN_DIR + 'Empty/*.jpg'

# Select model to train
model = TransferModel((HEIGHT,WIDTH,3))
#model = MyModel((HEIGHT,WIDTH,3))

# classes balance
occupied_samples = len(list(glob(TRAIN_DIR + 'Occupied/*.jpg')))
empty_samples = len(list(glob(TRAIN_DIR + 'Empty/*.jpg')))
n_samples  = empty_samples+occupied_samples
majority = max(occupied_samples,empty_samples)
neg_weight = round(float(majority/empty_samples),4)
pos_weight = round(float(majority/occupied_samples),4)
class_weight = {0: neg_weight, 1: pos_weight}
print('[INFO] class weight: {} ...'.format(class_weight))

aug =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range= 18,
      zoom_range = 0.15,
      width_shift_range = 0.2,
      height_shift_range = 0.2,
      shear_range = 0.15,
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest'
)
train_generator = aug.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, class_mode='binary')

valid_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = valid_datagen.flow_from_directory(VALID_DIR,target_size=(HEIGHT, WIDTH),batch_size=BATCH_SIZE,class_mode='binary')


opt = SGD(lr = LR, momentum=0.9, decay = LR/NB_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
history = model.fit(train_generator, steps_per_epoch=NB_TRAIN_SAMPLES // BATCH_SIZE \
      ,epochs=NB_EPOCHS, validation_data=validation_generator \
      , validation_steps=NB_VALID_SAMPLES // BATCH_SIZE, class_weight = class_weight\
      , verbose=1)


occupied_dir = VALID_DIR + 'Occupied/*.jpg'
empty_dir = VALID_DIR + 'Empty/*.jpg'
valid_images = list(glob(occupied_dir) + glob(empty_dir))
valid_images = valid_images[:NB_VALID_SAMPLES]

ground_truths = []
my_preds = []
for img in valid_images:
      img = img.replace("\\","/")
      label = img.split("/")[-2]
      if label == "Empty":
            ground_truths.append(0)
      else:
            ground_truths.append(1)
      image = cv2.imread(img)
      image = cv2.resize(image, (WIDTH, HEIGHT))
      image_x = np.expand_dims(image, axis=0)
      image_x = preprocess_input(image_x)
      pred = model.predict(image_x)
      pred = np.squeeze(pred)

      # print(pred)
      if pred > 0.8:
            my_preds.append(1)
      else:
            my_preds.append(0)
# my_preds = model.predict_generator(validation_generator, verbose=0)
# my_preds = list(np.argmax(my_preds, axis=-1))

print("[INFO] evaluating network...")
print(classification_report(ground_truths,my_preds, target_names=CLASSES))
print("confusion matrix...")
print(confusion_matrix(ground_truths,my_preds))


print("[INFO] Plotting...")

N = np.arange(0, NB_EPOCHS)

#title = "Training Loss and Accuracy on PKLOT-EXT baseline model"
title = "Training Loss and Accuracy on PKLOT-EXT Transfer model"
#title = "Training Loss and Accuracy on PKLOT-EXT Transfer-SVM"
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.plot(N, history.history["accuracy"], label="train_acc")
plt.plot(N, history.history["val_accuracy"], label="val_acc")
plt.title(title)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

if not os.path.isdir(os.path.join('imgs')):
      os.makedirs(os.path.join('imgs').replace("\\","/"))
      print('[INFO] imgs folder created at ', './imgs')
plt.savefig('./imgs/' + title)

# print("[INFO] Saving model...")
# model.save('./ .hdf5')