import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model

class TransferModel(Model):
      def __init__(self, input_shape,classes = 1, chanDim=-1):
            super(TransferModel, self).__init__()
            self.classes = classes

            self.base_model = VGG16(input_shape = input_shape, include_top=False, weights = 'imagenet')
            
            self.flatten = Flatten()
            self.dense = Dense(256, activation = 'relu')
            self.bn = BatchNormalization()

            self.out = Dense(classes)

            self.softmax = Activation("sigmoid")
            self.sigmoid = Activation("sigmoid")
      
      def call(self,inputs):
            self.base_model.trainable = False
            x = self.base_model(inputs, training=False)
            x = self.flatten(x)
            x = self.dense(x)
            x = self.bn(x)

            x = self.out(x)
            if self.classes > 1:
                  x = self.softmax(x)
            else:
                  x = self.sigmoid(x)
            return x