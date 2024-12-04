import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    
    #TODO: specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'relu':
        
       return np.maximum(0, x) 
   
    elif activation == 'sigmoid':
            
        return 1 / (1 + np.exp(-x))
    
    elif activation == 'softmax':
        
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    elif activation == 'linear':
            
           return x
           
    else:
        raise Exception("Activation function is not valid", activation) 

#-------------------------------
# Our own implementation of an MLP
#-------------------------------
class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation

        # TODO: specify the number of hidden layers based on the length of the provided lists
        self.hidden_layers = len(W) - 1

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = sum([i.size for i in W]) + sum([j.size for j in b])

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    def feedforward(
            self,
            x      # Input data points
        ):
            # TODO: specify a matrix for storing output values
            y = np.zeros((len(x), self.dataset.K))
            #input_layer = np.random.randn(2,1)

            # TODO: implement the feed-forward layer operations
            # Step 1: Initialize hidden layer
            h = 0
            
            for i, x_i in enumerate(x):
                # Step 2: Specify input layer (2x1 matrix)
                h = np.reshape(x_i, (2, 1))  # Assuming each input is 2-dimensional
                
                # Step 3: Loop over each hidden layer
                for j in range(self.hidden_layers):
                    h = np.dot(self.W[j], h) + self.b[j]
                    h = activation(h, self.activation)
                
                # Step 4: Final layer with softmax activation
                h = np.dot(self.W[-1], h) + self.b[-1]
                h = activation(h, 'softmax')
                
                # Store result for this data point
                y[i] = h.flatten()

            return y
            
    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class
        
        outputs = self.feedforward(self.dataset.x_train)        
        outputs_test = self.feedforward(self.dataset.x_test)
        
        train_loss = np.mean((outputs - self.dataset.y_train_oh)**2)
        train_acc = np.mean(np.argmax(outputs, 1) == self.dataset.y_train) * 100
        
        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the MLP
        test_loss = np.mean((outputs_test - self.dataset.y_test_oh)**2)
        test_acc = np.mean(np.argmax(outputs_test, 1) == self.dataset.y_test) * 100
        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)

