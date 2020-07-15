import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model

class MyModel(Model):
      def __init__(self, classes = 1, chanDim=-1):
            super(MyModel, self).__init__()
            self.classes = classes
            #self.pretrained = MobileNetV2(include_top=False, weights='imagenet')
            
            self.conv1 = Conv2D(64, (3, 3), padding="same", activation= 'relu')
            self.bn1 = BatchNormalization(axis=chanDim)

            self.conv2 = Conv2D(32, (3, 3), padding="same", activation= 'relu')
            self.bn2 = BatchNormalization(axis=chanDim)
            self.pool1 = MaxPooling2D(pool_size=(2, 2))

            self.conv2A = Conv2D(32, (3, 3), padding="same", activation='relu')
            self.bn2A = BatchNormalization(axis=chanDim)
            self.pool2 = MaxPooling2D(pool_size=(2, 2))

            self.flatten = Flatten()
            self.dense3 = Dense(64, activation = 'relu')
            self.bn3 = BatchNormalization()

            self.out = Dense(classes)
            self.sigmoid = Activation("sigmoid")
            self.softmax = Activation("softmax")
      
      def call(self, inputs):
            #x = self.pretrained(inputs)

            x = self.conv1(inputs)
            #x = self.conv1(x)
            x = self.bn1(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.bn2(x)

            x = self.conv2A(x)
            x = self.bn2A(x)
            x = self.pool2(x)

            x = self.flatten(x)
            x = self.dense3(x)
            x = self.bn3(x)

            x = self.out(x)

            if self.classes > 1:
                  x = self.softmax(x)
            else:
                  x = self.sigmoid(x)

            return x