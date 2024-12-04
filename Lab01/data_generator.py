import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Generate 2D synthetic data with class labels
def synthetic_data(
    N,                 # Number of points for each class
    K = 2,             # Number of classes
    stype = 'linear',  # 'linear' or 'polar' data
    sigma=0.2          # Standard deviation of normally distributed sample point locations
):
    x = np.zeros((K*N,2))
    y = np.zeros((K*N))

    # Linearly separable datapoints
    if stype == 'linear':
        for k in range(K):
            x[k*N:(k+1)*N,:] = np.random.normal((k+1)/(K+1),sigma,(N,2)) # Normally distributed 2D points
            y[k*N:(k+1)*N] = k # Class label

    # Radially separable datapoints
    elif stype == 'polar':
        xp = np.zeros((K*N,2))
        for k in range(K):
            xp[k*N:(k+1)*N,0] = np.abs(np.random.normal(k/K,sigma,N)) # Normally distributed radial location
            xp[k*N:(k+1)*N,1] = np.random.uniform(0,2*3.14,N) # Uniformly distributed angular location
            y[k*N:(k+1)*N] = k
        x[:,0] = xp[:,0]*np.cos(xp[:,1]) # Polar -> cartesian, dimension 1
        x[:,1] = xp[:,0]*np.sin(xp[:,1]) # Polar -> cartesian, dimension 2
    else:
        raise Exception("Dataset type is not valid", stype) 
    
    return x, y


#-------------------------------
# Data generator class
#-------------------------------
class DataGenerator:
    def __init__(self, verbose=True):
        self.verbose = verbose

    # Generate training, validation, and testing data
    def generate(
        self, 
        dataset='linear',   # Dataset type
        N_train=32,         # Number of samples in each class for training data
        N_test=512,         # Number of samples in each class for test data
        K=2,                # Number of classes
        sigma=0.05          # Standard deviation of normally distributed sample point locations
    ):
        self.N_train = N_train
        self.N_test = N_test
        self.K = K
        self.dataset = dataset
        self.sigma = sigma

        # Generate datasets
        self.x_train, self.y_train = synthetic_data(N=N_train, K=K, stype=dataset, sigma=sigma)
        self.x_valid, self.y_valid = synthetic_data(N=N_test, K=K, stype=dataset, sigma=sigma)
        self.x_test, self.y_test = synthetic_data(N=N_test, K=K, stype=dataset, sigma=sigma)

        if self.verbose:
            print('Data specification:')
            print('\tDataset type:          ', self.dataset)
            print('\tNumber of classes:     ', self.K)
            print('\tStd of classes:        ', self.sigma)
            print('\tTraining data shape:   ', self.x_train.shape)
            print('\tValidation data shape: ', self.x_valid.shape)
            print('\tTest data shape:       ', self.x_test.shape)

        # Convert labels to one-hot encoded vectors
        self.y_train_oh = keras.utils.to_categorical(self.y_train, K)
        self.y_valid_oh = keras.utils.to_categorical(self.y_valid, K)
        self.y_test_oh = keras.utils.to_categorical(self.y_test, K)

    # Plotting of training and testing data
    def plot(
        self,
        save_path=None  # Specify a filename if you want to export the plot
    ):
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        scatter = plt.scatter(self.x_train[:,0], self.x_train[:,1], s=20, c=self.y_train, alpha=0.8)
        plt.title('Training data')
        plt.axis('equal')
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=['Class %02d'%(i+1) for i in range(self.K)], title="Class labels")
        
        plt.subplot(1,2,2)
        scatter = plt.scatter(self.x_test[:,0], self.x_test[:,1], s=20, c=self.y_test, alpha=0.8)
        plt.title('Test data')
        plt.axis('equal')
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=['Class %02d'%(i+1) for i in range(self.K)], title="Class labels")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    # For sampling the input space of a model
    def sample_input_space(
        self,
        model,    # Model to sample
        G,        # Resolution for sampling input space
        mi = 0,   # Minimum input
        ma = 1    # Maximum input
    ):
        # Grid with combinations of inputs
        xv, yv = np.meshgrid(np.linspace(mi,ma,G),np.linspace(mi,ma,G))
        xv = np.reshape(xv,G*G)
        yv = np.reshape(yv,G*G)
        x_grid = np.concatenate((xv[:,np.newaxis], yv[:,np.newaxis]), 1)

        yp_test = model.feedforward(self.x_test) # Output on test set
        yp_test = np.argmax(yp_test,1)           # Predicted class is the output with maximum value

        yp_grid = model.feedforward(x_grid) # Output on the grid with input combinations
        yp_grid = np.argmax(yp_grid,1)      # Predicted class is the output with maximum value

        return yp_grid, yp_test

    # Plotting of training and testing data with decision boundary of a provided model
    def plot_classifier(self,
                        model,          # The model we want to analyze (KerasMLP or MLP)
                        G=128,          # Grid resolution for drawing decision boundaries
                        save_path=None  # Specify a filename if you want to export the plot
                       ):
        # Sampling grid, for evaluating the model at different combinations of inputs
        mi, ma = np.min(self.x_test)-0.1, np.max(self.x_test)+0.1    # Boundaries of the input space
        yp_grid, yp_test = self.sample_input_space(model, G, mi, ma) # Sample the model
        msk = (yp_test!=self.y_test)                                 # Indices of misclassified points

        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.imshow(np.flip(np.reshape(yp_grid,(G,G)),0),
                   extent=[mi, ma, mi, ma], alpha=0.1) # The sampled grid
        scatter = plt.scatter(self.x_train[:,0], self.x_train[:,1],
                              c=self.y_train, s=15, alpha=0.9) # Training data points
        plt.title('Training data')
        plt.axis('equal')
        plt.legend(handles=scatter.legend_elements()[0], loc='upper left',
                   labels=['Class %02d'%(i+1) for i in range(self.K)])
        plt.tight_layout()

        plt.subplot(1,2,2)
        plt.imshow(np.flip(np.reshape(yp_grid,(G,G)),0),
                   extent=[mi, ma, mi, ma], alpha=0.1) # The sampled grid
        scatter1 = plt.scatter(self.x_test[:,0], self.x_test[:,1],
                               c=self.y_test, s=15, alpha=0.9) # Test data points
        scatter2 = plt.scatter(self.x_test[msk,0], self.x_test[msk,1],
                               facecolors='none', edgecolors='r',s=50,alpha=1) # Misclassified points
        plt.title('Test data')
        plt.axis('equal')
        plt.legend(handles=scatter1.legend_elements()[0]+[scatter2], loc='upper left',
                   labels=['Class %02d'%(i+1) for i in range(self.K)] + ['Misclassified points'])
        plt.xlim(mi,ma)
        plt.ylim(mi,ma)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
