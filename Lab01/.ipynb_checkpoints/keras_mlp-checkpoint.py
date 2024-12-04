from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import data_generator

#-------------------------------
# Keras wrapper for an MLP
#-------------------------------
class KerasMLP:
    def __init__(
        self,
        dataset,         # DataGenerator
        verbose=True     # For printing info messages
    ):
        self.verbose = verbose
        self.dataset = dataset
    
    # Keras MLP model specification
    def setup_model(
        self,
        hidden_layers=0,      # Number of hidden layers (total number of layers will be L=hidden_layers+1)
        layer_width=2,        # Number of neurons in each hidden layers
        activation='linear',  # Activation function
        init='normal'         # Initialization method
    ):
        self.hidden_layers = hidden_layers
        self.layer_width = layer_width
        keras.backend.clear_session()  # Clear memory

        self.model = keras.Sequential() # Feed-forward model
        self.model.add(keras.layers.InputLayer(input_shape=(2,))) # 2-dimensional input layer

        # The hidden layers
        for i in range(hidden_layers):
            # Fully-connected layer with 'layer_width' number of neurons
            self.model.add(keras.layers.Dense(layer_width,
                                              activation=activation, 
                                              kernel_initializer=init))
        # Output layer, with softmax layer for classification
        self.model.add(keras.layers.Dense(self.dataset.K,
                                          activation='softmax',
                                          kernel_initializer=init))
        
        if self.verbose:
            self.model.summary()  # Model info

    # Compile model with loss function and optimizer
    def compile(
        self, 
        loss_fn=keras.losses.CategoricalCrossentropy(),   # Cross entropy loss by default
        optimizer=keras.optimizers.SGD(learning_rate=1.0)   # Vanilla SGD by default
    ): 
        self.model.compile(loss=loss_fn,
                           optimizer=optimizer,
                           metrics=['accuracy'])  # Monitor the accuracy

    # Optimization of model
    def train(
        self,
        batch_size=512,  # Batch size for SGD 
        epochs=100      # Number of training epochs
    ):
        # Training loop
        self.log = self.model.fit(self.dataset.x_train, self.dataset.y_train_oh,
                                  batch_size=batch_size, epochs=epochs,
                                  validation_data=(self.dataset.x_valid, self.dataset.y_valid_oh), # Monitor on validation set
                                  shuffle=True, verbose=self.verbose) # Shuffle randomizes order of data points each epoch

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # Provides loss and accuracy on the training set
        score = self.model.evaluate(self.dataset.x_train, self.dataset.y_train_oh, verbose=self.verbose)
        print('\tTrain loss:     %0.4f'%score[0])
        print('\tTrain accuracy: %0.2f'%(100*score[1]))

        # Provides loss and accuracy on the test set
        score = self.model.evaluate(self.dataset.x_test, self.dataset.y_test_oh, verbose=self.verbose)
        print('\tTest loss:      %0.4f'%score[0])
        print('\tTest accuracy:  %0.2f'%(100*score[1]))

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      # Input data points
    ):
        return self.model.predict(x, verbose=self.verbose) # Keras model feed-forward function

    # Plot training log
    def plot_training(self, save_path=None):
        plt.figure(figsize=(18,4))

        # Plot loss on training and validation set
        plt.subplot(1,2,1)
        plt.plot(self.log.history['loss'])
        plt.plot(self.log.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid('on')
        plt.legend(['Train', 'Validation'])

        # Plot accuracy on training and validation set
        plt.subplot(1,2,2)
        plt.plot(self.log.history['accuracy'])
        plt.plot(self.log.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.grid('on')
        plt.legend(['Train', 'Validation'])

        # Export figure if a path is provided
        if save_path:
            plt.savefig(save_path)
        plt.show()

    # Extract weights and biases from model
    def get_weights(self):
        # Use only the fully-connected (dense) layers (there could be other layers specified)
        layers = [l for l in self.model.layers if l.name.find('dense') == 0]
        W = []
        b = []

        # Extract weight matrices and bias vectors
        for l in range(len(layers)):
            Wl, bl = layers[l].get_weights()
            W.append(np.transpose(Wl))
            b.append(bl[:,np.newaxis])
            
        return W, b
