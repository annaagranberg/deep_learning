import numpy as np
import h5py
from tensorflow import keras
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

#-------------------------------
# Data generator class
#-------------------------------
class DataGenerator:
    def __init__(self, verbose=True):
        self.verbose = verbose

    # Generate training, validation, and testing data
    def generate(
        self, 
        dataset='mnist',   # Dataset type
        N_train=None,      # Number of training samples (if not specified, all samples will be used)
        N_valid=0.1        # Fraction of training samples to use as validation data
    ):
        self.N_train = N_train
        self.N_valid = N_valid
        self.dataset = dataset


        # CIFAR10 dataset, provided through Keras
        if dataset == 'cifar10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()
            self.split_data()
            self.normalize()
            self.x_train_gray = self.prepare_data(self.x_train)
            self.x_test_gray = self.prepare_data(self.x_test)
            self.x_valid_gray = self.prepare_data(self.x_valid)


        # Number of classes
        self.K = len(np.unique(self.y_train))
        
        # Number of color channels
        self.C = self.x_train.shape[3]
        
        # One hot encoding of class labels
        self.y_train_oh = keras.utils.to_categorical(self.y_train, self.K)
        self.y_valid_oh = keras.utils.to_categorical(self.y_valid, self.K)
        self.y_test_oh = keras.utils.to_categorical(self.y_test, self.K)

        if self.verbose:
            print('Data specification:')
            print('\tDataset type:          ', self.dataset)
            print('\tNumber of classes:     ', self.K)
            print('\tNumber of channels:    ', self.C)
            print('\tTraining data shape:   ', self.x_train.shape)
            print('\tValidation data shape: ', self.x_valid.shape)
            print('\tTest data shape:       ', self.x_test.shape)

    # Training/validation data split
    def split_data(self):
        # Random shuffle of training data
        N = self.x_train.shape[0]
        ind = np.random.permutation(N)
        self.x_train = self.x_train[ind]
        self.y_train = self.y_train[ind]

        # Validation data as subset of training data
        self.N_valid = int(N*self.N_valid)
        N = N - self.N_valid
        self.x_valid = self.x_train[-self.N_valid:]
        self.y_valid = self.y_train[-self.N_valid:]

        # Subset of training data, if specified
        if self.N_train and self.N_train < N:
            self.x_train = self.x_train[:self.N_train]
            self.y_train = self.y_train[:self.N_train]
        else:
            self.x_train = self.x_train[:N]
            self.y_train = self.y_train[:N]
            self.N_train = N

    # Normalization of images, from 8-bit images in the range [0,255]
    # to floating point representation in the range [-1,1]
    def normalize(self):
        # self.x_train = 2*self.x_train.astype("float32") / 255 - 1.0
        # self.x_valid = 2*self.x_valid.astype("float32") / 255 - 1.0
        # self.x_test = 2*self.x_test.astype("float32") / 255 - 1.0
        self.x_train = self.x_train.astype("float32") / 255.0  # Range [0, 1]
        self.x_valid = self.x_valid.astype("float32") / 255.0
        self.x_test = self.x_test.astype("float32") / 255.0

        
    def prepare_data(self, x_color):
        """
        Converts color images to grayscale for input and keeps the color images for output.
        """
        x_gray = np.expand_dims(rgb2gray(x_color), axis=-1)  # Add channel dimension for grayscale
        return x_gray
        
    # Show some training samples
    def plot(self, num_samples=12, save_path=None):
        """
        Visualize a subset of training data: grayscale input and color output.
        """
        if self.x_train_gray is None or self.x_train is None:
            raise ValueError("Data not generated yet. Call `generate()` first.")

        cols = num_samples
        rows = 2  # Grayscale and color for each sample
        plt.figure(figsize=(cols * 2, rows * 2))

        for i in range(num_samples):
            # Grayscale input
            plt.subplot(rows, cols, i + 1)
            plt.imshow(np.clip(self.x_train_gray[i].squeeze(), 0, 1), cmap='gray')
            plt.title("Input (Grayscale)")
            plt.axis('off')

            # Color output
            plt.subplot(rows, cols, i + cols + 1)
            plt.imshow(self.x_train[i])
            plt.title("Target (Color)")
            plt.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

