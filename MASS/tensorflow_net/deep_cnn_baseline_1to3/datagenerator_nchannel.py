# a data generator to handle multichannel data
import numpy as np
from scipy.io import loadmat
import h5py

class DataGeneratorNChannel:
    def __init__(self, data_shape=np.array([29, 129, 1]), shuffle=False, test_mode=False):

        # Init params
        self.shuffle = shuffle
        self.data_shape = data_shape
        self.pointer = 0
        self.X = np.array([])
        self.y = np.array([])
        self.label = np.array([])
        self.boundary_index = np.array([])
        self.test_mode = test_mode

        self.Ncat = 5
        # read from mat file

        if self.shuffle:
            self.shuffle_data()

    def indexing(self):
        print("Boundary indices")
        print(self.boundary_index)

        self.data_size = len(self.label)
        self.data_index = np.arange(self.data_size)
        print(len(self.data_index))
        if self.test_mode == False:
            mask = np.in1d(self.data_index,self.boundary_index, invert=True)
            self.data_index = self.data_index[mask]
            print(len(self.data_index))
        print(self.X.shape, self.y.shape, self.label.shape)
        
    def shuffle_data(self):
        """
        Random shuffle the data points indexes
        """
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(self.data_index))
        self.data_index = self.data_index[idx]

                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) samples and labels
        """
        data_index = self.data_index[self.pointer : self.pointer + batch_size]

        #update pointer
        self.pointer += batch_size

        batch_x = np.ndarray([batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]])
        batch_y1 = np.ndarray([batch_size, self.y.shape[1]])
        batch_y2 = np.ndarray([batch_size, self.y.shape[1]])
        batch_y3 = np.ndarray([batch_size, self.y.shape[1]])
        batch_label1 = np.ndarray([batch_size])
        batch_label2 = np.ndarray([batch_size])
        batch_label3 = np.ndarray([batch_size])

        for i in range(len(data_index)):
            batch_x[i] = self.X[data_index[i], :, :, :]
            batch_y2[i] = self.y[data_index[i]]
            batch_label2[i] = self.label[data_index[i]]
            if(self.test_mode == True and data_index[i] == 0):
                batch_y1[i] = self.y[data_index[i]]
                batch_label1[i] = self.label[data_index[i]]
            else:
                batch_y1[i] = self.y[data_index[i] - 1]
                batch_label1[i] = self.label[data_index[i] -1]
            batch_y3[i] = self.y[data_index[i] + 1]
            batch_label3[i] = self.label[data_index[i] + 1]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        # Get next batch of image (path) and labels
        batch_x.astype(np.float32)
        batch_y1.astype(np.float32)
        batch_label1.astype(np.float32)
        batch_y2.astype(np.float32)
        batch_label2.astype(np.float32)
        batch_y3.astype(np.float32)
        batch_label3.astype(np.float32)

        #return array of images and labels
        return batch_x, batch_y1, batch_label1, batch_y2, batch_label2, batch_y3, batch_label3

    # get and padding zeros the rest of the data if they are smaller than 1 batch
    # this necessary for testing
    def rest_batch(self, batch_size):

        #data_index = self.data_index[self.pointer : self.data_size]
        data_index = self.data_index[self.pointer : self.data_size]
        actual_len = self.data_size - self.pointer

        # update pointer
        self.pointer = self.data_size

        batch_x = np.ndarray([actual_len, self.data_shape[0], self.data_shape[1], self.data_shape[2]])
        batch_y1 = np.ndarray([actual_len, self.y.shape[1]])
        batch_y2 = np.ndarray([actual_len, self.y.shape[1]])
        batch_y3 = np.ndarray([actual_len, self.y.shape[1]])
        batch_label1 = np.ndarray([actual_len])
        batch_label2 = np.ndarray([actual_len])
        batch_label3 = np.ndarray([actual_len])


        for i in range(len(data_index)):
            batch_x[i] = self.X[data_index[i], :, :, :]
            batch_y2[i] = self.y[data_index[i]]
            batch_label2[i] = self.label[data_index[i]]
            batch_y1[i] = self.y[data_index[i] - 1]
            batch_label1[i] = self.label[data_index[i] - 1]
            if(self.test_mode == True and data_index[i] == self.data_size - 1):
                batch_y3[i] = self.y[data_index[i]]
                batch_label3[i] = self.label[data_index[i]]
            else:
                batch_y3[i] = self.y[data_index[i] + 1]
                batch_label3[i] = self.label[data_index[i] + 1]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        # Get next batch of image (path) and labels
        batch_x.astype(np.float32)
        batch_y1.astype(np.float32)
        batch_label1.astype(np.float32)
        batch_y2.astype(np.float32)
        batch_label2.astype(np.float32)
        batch_y3.astype(np.float32)
        batch_label3.astype(np.float32)

        # return array of images and labels
        return actual_len, batch_x, batch_y1, batch_label1, batch_y2, batch_label2, batch_y3, batch_label3

