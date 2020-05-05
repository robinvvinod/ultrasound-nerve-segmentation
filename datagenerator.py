# yapf: disable
from keras.utils import Sequence
from keras.layers.convolutional import ZeroPadding2D
import numpy as np
from PIL import Image

class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels=[], batch_size=1, dim=(512,512,512), n_channels=1, n_classes=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Creates an empty placeholder array that will be populated with data that is to be supplied
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, ID in enumerate(list_IDs_temp):
            im = np.array(Image.open(ID))
            gt = np.array(Image.open(ID.replace(".tif", "_mask.tif")))

            # Converting into binary array of 0s and 1s
            im = np.where(im == 255, 1, 0)
            gt = np.where(gt == 255, 1, 0)
            
            im = np.pad(im, ( (30,30),(30,30) ), 'constant')
            gt = np.pad(gt, ( (30,30),(30,30) ), 'constant')

            im = np.reshape(im, (480,640,1))
            gt = np.reshape(gt, (480,640,1))

            X[i, ] = im
            y[i, ] = gt

        return X, y
