import numpy as np
from scipy.io import loadmat
import h5py

class EqualDataGenerator:
    def __init__(self, filelist, data_shape=np.array([3000]),shuffle=False):

        # Init params
        self.shuffle = shuffle
        self.filelist = filelist
        self.data_shape = data_shape
        self.X = np.array([])
        self.y = np.array([])
        self.label = np.array([])
        self.boundary_index = np.array([])

        self.Ncat = 5
        # read from mat file

        self.read_mat_filelist(self.filelist)

        self.data_size = len(self.label)
        # create pointers for different classes
        self.nclass = self.y.shape[1]
        self.data_index = []
        for i in range(self.nclass):
            ind = np.where(self.y[:,i] == 1)[0]
            print(len(ind))
            mask = np.in1d(ind,self.boundary_index, invert=True)
            ind = ind[mask]
            print(len(ind))
            self.data_index.append(ind)

        self.pointer = np.zeros([self.nclass,1])

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
                #print(l)
                items = l.split()
                files.append(items[0])
                self.data_size += int(items[1])
                print(self.data_size)
        self.X = np.ndarray([self.data_size, self.data_shape[0]])
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

        print("Boundary indices")
        print(self.boundary_index)
        print(self.X.shape, self.y.shape, self.label.shape)

    def read_mat_file(self,filename):
        """
        Read matfile HD5F file and parsing
        """
        # Load data
        print(filename)
        data = h5py.File(filename,'r')
        data.keys()
        X = np.array(data['X'])
        X = np.transpose(X, (1, 0))  # rearrange dimension

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
        #create list of permutated index and shuffle data accoding to list

        for i in range(self.nclass):
            data_index = self.data_index[i]
            idx = np.random.permutation(len(data_index))
            data_index = data_index[idx]
            self.data_index[i] = data_index


    def numel_per_class(self, classid):
        if(classid >= 0 and classid < self.nclass):
            return len(self.data_index[classid])
        else:
            return 0
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        for i in range(self.nclass):
            self.pointer[i] = 0
        
        if self.shuffle:
            self.shuffle_data()

    def next_batch_per_class(self, batch_size, classid):
        """
        This function gets the next n ( = batch_size) sample from a class
        """
        class_size = self.numel_per_class(classid)
        if(self.pointer[classid] + batch_size <= class_size):
            data_index = self.data_index[classid][int(self.pointer.item(classid)):int(self.pointer.item(classid)) + batch_size]
            self.pointer[classid] += batch_size #update pointer
        else:
            data_index = self.data_index[classid][int(self.pointer.item(classid)): class_size]
            leftover = batch_size - (class_size - self.pointer.item(classid))
            data_index = np.concatenate([data_index, self.data_index[classid][0 : int(leftover)]])
            self.pointer[classid] = leftover    #update pointer
        return data_index

    
    def next_batch(self, batch_size_per_class):
        """
        This function gets the next n ( = batch_size) sample from every class
        """
        data_index = []
        for i in range(self.nclass):
            data_index_i = self.next_batch_per_class(batch_size_per_class, i)
            data_index = np.concatenate([data_index, data_index_i])
        idx = np.random.permutation(len(data_index))
        data_index = data_index[idx]

        batch_size = batch_size_per_class*self.nclass
        batch_x = np.ndarray([3*batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]]) # account for 3-epoch contexutual input
        batch_y = np.ndarray([batch_size, self.y.shape[1]])
        batch_label = np.ndarray([batch_size])

        for i in range(len(data_index)):
            batch_y[i] = self.y[int(data_index.item(i))]
            batch_label[i] = self.label[int(data_index.item(i))]

            batch_x[i*3] = self.X[int(data_index[i] - 1)]
            batch_x[i*3+1] = self.X[int(data_index[i])]
            batch_x[i*3+2] = self.X[int(data_index[i] + 1)]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        batch_x.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        return batch_x, batch_y, batch_label
