# load data from mat files and generate batch-by-batch
import numpy as np
from scipy.io import loadmat
import h5py

class DataGenerator:
    def __init__(self, filelist, data_shape=np.array([29, 129]),shuffle=False, test_mode=False):

        # Init params
        self.shuffle = shuffle
        self.filelist = filelist
        self.data_shape = data_shape
        self.pointer = 0
        self.X = np.array([])
        self.y = np.array([])
        self.label = np.array([])
        self.boundary_index = np.array([])
        self.test_mode = test_mode

        self.Ncat = 5 # sorry for the hard code here
        # read from mat file

        self.read_mat_filelist(self.filelist)
        
        if self.shuffle:
            self.shuffle_data()

    def read_mat_filelist(self,filelist):
        """
        Scan the file list and read them one-by-one
        """
        files = []
        self.data_size = 0
        with open(filelist) as f:
            lines = f.readlines()
            for l in lines:
                items = l.split()
                files.append(items[0])
                self.data_size += int(items[1])
                print(self.data_size)
        self.X = np.ndarray([self.data_size, self.data_shape[0], self.data_shape[1]])
        self.y = np.ndarray([self.data_size, self.Ncat])
        self.label = np.ndarray([self.data_size])
        count = 0
        for i in range(len(files)):
            X, y, label = self.read_mat_file(files[i].strip())
            self.X[count : count + len(X)] = X
            self.y[count : count + len(X)] = y
            self.label[count : count + len(X)] = label
            self.boundary_index = np.append(self.boundary_index, [count, (count + len(X) - 1)])
            count += len(X)
            print(count)

        # exclude boundary indexes (terminal epochs) from the data indexes
        # however, if test_mode=True (this generate data batches for testing), boundary indexes
        # will be actually included (see next_batch function)
        print("Boundary indices")
        print(self.boundary_index)
        self.data_index = np.arange(len(self.X))
        print(len(self.data_index))
        if self.test_mode == False:
            mask = np.in1d(self.data_index,self.boundary_index, invert=True)
            self.data_index = self.data_index[mask]
            print(len(self.data_index))
        print(self.X.shape, self.y.shape, self.label.shape)
        #self.data_size = len(self.label)

    def read_mat_file(self,filename):
        """
        Read matfile HD5F file and parsing
        """
        # Load data
        print(filename)
        data = h5py.File(filename,'r')
        data.keys()
        X = np.array(data['X'])
        X = np.transpose(X, (2, 1, 0))  # rearrange dimension

        y = np.array(data['y'])
        y = np.transpose(y, (1, 0))  # rearrange dimension
        label = np.array(data['label'])
        label = np.transpose(label, (1, 0))  # rearrange dimension
        label = np.squeeze(label)

        return X, y, label

    def shuffle_data(self):
        """
        Random shuffle the data points indexes
        """
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
        data_index = self.data_index[self.pointer:self.pointer + batch_size]

        #update pointer
        self.pointer += batch_size

        batch_x = np.ndarray([batch_size, self.data_shape[0], self.data_shape[1]])
        batch_y1 = np.ndarray([batch_size, self.y.shape[1]])
        batch_y2 = np.ndarray([batch_size, self.y.shape[1]])
        batch_y3 = np.ndarray([batch_size, self.y.shape[1]])
        batch_label1 = np.ndarray([batch_size])
        batch_label2 = np.ndarray([batch_size])
        batch_label3 = np.ndarray([batch_size])

        for i in range(len(data_index)):
            batch_x[i] = self.X[data_index[i], :, :]
            batch_y2[i] = self.y[data_index[i]]
            batch_label2[i] = self.label[data_index[i]]
            if(self.test_mode == True and data_index[i] == 0): # padding the terminal epochs when if test_mode == True
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

    # get the rest of the data if they are smaller than one regular batch
    # this necessary for testing
    def rest_batch(self, batch_size):

        data_index = self.data_index[self.pointer:self.data_size]
        actual_len = self.data_size - self.pointer

        # update pointer
        self.pointer = self.data_size

        batch_x = np.ndarray([actual_len, self.data_shape[0], self.data_shape[1]])
        batch_y1 = np.ndarray([actual_len, self.y.shape[1]])
        batch_y2 = np.ndarray([actual_len, self.y.shape[1]])
        batch_y3 = np.ndarray([actual_len, self.y.shape[1]])
        batch_label1 = np.ndarray([actual_len])
        batch_label2 = np.ndarray([actual_len])
        batch_label3 = np.ndarray([actual_len])


        for i in range(len(data_index)):
            batch_x[i] = self.X[data_index[i], :, :]
            batch_y2[i] = self.y[data_index[i]]
            batch_label2[i] = self.label[data_index[i]]
            batch_y1[i] = self.y[data_index[i] - 1]
            batch_label1[i] = self.label[data_index[i] - 1]
            if(self.test_mode == True and data_index[i] == self.data_size - 1): # padding the terminal epochs when if test_mode == True
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


    def filter_with_filterbank(self, Wfb):
        X = np.reshape(self.X, (self.data_size*self.data_shape[0], self.data_shape[1]))
        X = np.dot(X, Wfb)
        self.X = np.reshape(X, (self.data_size, self.data_shape[0], Wfb.shape[1]))

        self.data_shape = self.X.shape[1:]
        del X
