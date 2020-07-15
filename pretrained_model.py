import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model

class TransferModel(Model):
      def __init__(self, classes = 1, chanDim=-1):
            super(TransferModel, self).__init__()
            self.classes = classes

            self.pretrained = ResNet50(include_top=False, weights='imagenet')
            self.bn1 = BatchNormalization(axis=chanDim)

            self.flatten = Flatten()
            self.dense3 = Dense(64, activation = 'relu')
            self.bn3 = BatchNormalization()

            self.out = Dense(classes)
            
            self.sigmoid = Activation("sigmoid")
            self.softmax = Activation("softmax")
      
      def call(self, inputs):
            #x = self.pretrained(inputs)
            for layer in self.pretrained.layers:
                  layer.trainable = False
            
            x = self.pretrained.output
            x = self.bn1(x)

            x = self.flatten(x)
            x = self.dense3(x)
            x = self.bn3(x)

            x = self.out(x)

            if self.classes > 1:
                  x = self.softmax(x)
            else:
                  x = self.sigmoid(x)

            return x