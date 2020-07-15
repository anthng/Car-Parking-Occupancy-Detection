import os
from glob import glob

import tensorflow as tf

from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import linear_model


TRAIN_DIR = './dataset/PKLot/custom_dataset/train/'
VALID_DIR = './dataset/PKLot/custom_dataset/valid/'
ROOT_DIR = '../../dataset/Pklot/PKLotSegmented/'
CLASSES = ['Empty', 'Occupied']

def load_dataset(path):
      X = []
      features = list(glob(path + '*.npy'))

      for f in features:
            #print(f)
            data = np.load(f)
            #print(data)
            #print(data.shape)
            data = data.reshape(data.shape[1],)
            #print(data)
            X.append(data)
      X = np.array(X)
      return X

if __name__ == "__main__":
      X_train = load_dataset('./features/PKLot/Train/')
      y_train = np.load('./features/PKLot/train_label.npy')

      X_test = load_dataset('./features/PKLot/Valid/')
      y_test = np.load('./features/PKLot/valid_label.npy')
      
      print("[INFO] training data...")
      #clf = svm.SVC(C = 0.01, class_weight='balanced')
      clf = linear_model.SGDClassifier(loss = 'log',alpha = 1e-4,max_iter=1000, tol=1e-4, class_weight = 'balanced', n_jobs =-1)

      clf.fit(X_train, y_train)

      y_preds = clf.predict_proba(X_test)

      my_preds = []
      for pred in y_preds:
            if pred[1] > 0.95:
                  my_preds.append(1)
            else:
                  my_preds.append(0)
      print("[INFO] evaluating network...")
      print(classification_report(y_test,my_preds, target_names=CLASSES))
      print("confusion matrix...")
      print(confusion_matrix(y_test,my_preds))