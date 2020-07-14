import os
from glob import glob

import numpy as np
import pandas as pd

from shutil import copy2



ROOT_DIR = './dataset/Pklot/PKLotSegmented/'

labels = ["Empty", "Occupied"]

datasets = {
    'train': ["UFPR04/Sunny/", "UFPR04/Rainy/", "UFPR04/Cloudy/", "UFPR05/Sunny/", "UFPR05/Rainy/", "UFPR05/Cloudy/"],
    'valid': ["PUC/Sunny/","PUC/Rainy/", "PUC/Cloudy/"]
}

class DatasetUtils(object):
    def __init__(self, labels, datasets, to_csv_folder, train_size = 0.2, test_size = 0.05, random_state = 42):
        """
        Dataset construction:
            labels (list): labels in dataset (empty or occupied)
            datasets (dict): structure of dataset
            to_csv_folder (str): given a folder path to create csv file for training model.
            train_size (float): size of samples in dataset for train (Default 0.2)
            test_size (float): size of samples in dataset for test (Default 0.05)
            random_state (int): rule of random when shuffling dataset for training and validation. (Default 42)
        """

        self.labels = labels
        self.datasets = datasets
        self.to_csv_dir = to_csv_folder
        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state

        self.train = list()
        self.valid = list()

        self.__run__()
    
    def __run__(self):
        for label in labels:
            for dtype in self.datasets:
                for folder in self.datasets[dtype]:
                    dir_content = [d for d in os.listdir(os.path.join(ROOT_DIR,folder)) if os.path.isdir(os.path.join(ROOT_DIR,folder,d))]
                    #print(dir_content)
                    for dir in dir_content:
                        folder_path = os.path.join(ROOT_DIR,folder, dir, label)
                        images = glob(os.path.join(folder_path, "*.jpg"))
                        #print(images)
                        if len(images)>0:
                            if dtype == 'train':
                                sample_size = self.train_size
                            else:
                                sample_size = self.test_size
                            random_sample = np.random.choice(images, replace=False,size=int(len(images)*sample_size))

                            for img in random_sample:
                                #normalize \\ in path in windows
                                img = img.replace("\\","/")
                                image_name = img.split('/')[-1]
                                temp = {}
                                temp['image_name'] = image_name
                                temp["label"] = label
                                temp['folder_path'] = os.path.join(folder,dir)
                                temp['data_type'] = dtype
                                if dtype == 'train':
                                    self.train.append(temp)
                                else:
                                    self.valid.append(temp)
        
        df_train = pd.DataFrame(self.train).sample(frac = 1, random_state=self.random_state)
        df_valid = pd.DataFrame(self.valid).sample(frac = 1, random_state=self.random_state)

        print("Size of training: ", df_train.shape)
        print("Size of validation: ", df_valid.shape)

        print("Save to csv")
        df_train.to_csv(self.to_csv_dir + 'train.csv', index = False)
        df_valid.to_csv(self.to_csv_dir + 'valid.csv', index = False)

        print("Train dataframe")
        print(df_train.head(5))
        print("test dataframe")
        print(df_valid.head(5))

def custom_dataset(df, ROOT_DIR, path_folder, dtype = 'train', sample_size = -1):
    if not os.path.isdir(os.path.join(path_folder, dtype, 'Occupied')):
        os.makedirs(os.path.join(path_folder, dtype, 'Occupied').replace("\\","/"))
        print(dtype + '/Occupied folder created at ', path_folder)
    
    if not os.path.isdir(os.path.join(path_folder, dtype, 'Empty')):
        os.makedirs(os.path.join(path_folder, dtype, 'Empty').replace("\\","/"))
        print(dtype + '/Empty folder created at ', path_folder)

    for idx in df.index.values[:sample_size]:
        src = os.path.join(ROOT_DIR, df.loc[idx, 'folder_path'], df.loc[idx, 'label'], df.loc[idx, 'image_name']).replace("\\","/")
        if df.loc[idx, 'label'] == 'Occupied':
            dst = os.path.join(path_folder, dtype, 'Occupied').replace("\\","/")
        else:
            dst = os.path.join(path_folder, dtype, 'Empty').replace("\\","/")
        copy2(src, dst)

if __name__ == "__main__":
    dataset = DatasetUtils(labels, datasets, './dataset/PKLot/', train_size= 0.1/4, test_size= 0.05/8)
    
    df_train = pd.read_csv("./dataset/PKLot/train.csv")
    df_valid = pd.read_csv("./dataset/PKLot/valid.csv")

    path_folder = './dataset/PKLot/custom_dataset'

    if not os.path.isdir(path_folder):
        os.makedirs(path_folder)
        print('The dataset folder created at ', path_folder)

    # set sample size to create sample dataset
    train_sample_size = 5000
    valid_sample_size = 500

    custom_dataset(df = df_train, ROOT_DIR = ROOT_DIR, path_folder = path_folder, dtype = 'train',sample_size = train_sample_size)
    custom_dataset(df = df_valid, ROOT_DIR = ROOT_DIR, path_folder = path_folder, dtype = 'valid',sample_size = valid_sample_size)
