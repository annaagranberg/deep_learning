import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# Model evaluation with Keras model
def evaluate(model, dataset, final=False):
    print('Model performance:')
    
    # Provides loss and accuracy on the training set
    score = model.evaluate(dataset.x_train, dataset.y_train_oh, verbose=False)
    print('\tTrain loss:          %0.4f'%score[0])
    print('\tTrain accuracy:      %0.2f'%(100*score[1]))

    # If there are more metrics, we assume AUC
    if len(score) > 2:
        print('\tTrain AUC:           %0.2f'%(score[2]))

    # Provides loss and accuracy on the test set
    if final:
        score = model.evaluate(dataset.x_test, dataset.y_test_oh, verbose=False)
        print('\tTest loss:           %0.4f'%score[0])
        print('\tTest accuracy:       %0.2f'%(100*score[1]))
        
        if len(score) > 2:
            print('\tTest AUC:            %0.2f'%(score[2]))

    # Provides loss and accuracy on the validation set
    else:
        score = model.evaluate(dataset.x_valid, dataset.y_valid_oh, verbose=False)
        print('\tValidation loss:     %0.4f'%score[0])
        print('\tValidation accuracy: %0.2f'%(100*score[1]))

        if len(score) > 2:
            print('\tValidation AUC:     %0.2f'%(score[2]))

    return score

# Predictions on a test set without labels, exporting
# the results to a CSV file
def pred_test(model, dataset, name):
    logits = model.predict(dataset.x_test)
    pred = np.argmax(logits, axis=1)
    df = pd.DataFrame({'class': pred})
    df.index.name = 'id'
    df.to_csv(name)
    print('Done!Please upload your file to Kaggle!')
    return pred

# Plotting of training history
def plot_training(log):
    N_train = len(log.history['loss'])
    N_valid = len(log.history['val_loss'])
    
    plt.figure(figsize=(18,4))
    
    # Plot loss on training and validation set
    plt.subplot(1,2,1)
    plt.plot(log.history['loss'])
    plt.plot(np.linspace(0,N_train-1,N_valid), log.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid('on')
    plt.legend(['Train', 'Validation'])
    
    # Plot accuracy on training and validation set
    plt.subplot(1,2,2)
    plt.plot(log.history['accuracy'])
    plt.plot(np.linspace(0,N_train-1,N_valid), log.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid('on')
    plt.legend(['Train', 'Validation'])
    
    plt.show()

# Extraction of weights from Keras model
def get_weights(model):
    W = []
    b = []
    lname = []

    # Types of layers we want to extract
    layer_names = ['conv','pool','flatten','dense']
    
    # Extract weights and biases
    for l in range(len(model.layers)):
        for j in range(len(layer_names)):
            if model.layers[l].name.find(layer_names[j]) >= 0:
                lname.append(layer_names[j])
        Wl = model.layers[l].get_weights()

        # Convolutional kernels and biases for conv layers
        if lname[l] == 'conv':
            W.append(Wl[0])
            b.append(Wl[1])

        # Weight matrix and biases for dense layers
        elif lname[l] == 'dense':
            W.append(np.transpose(Wl[0]))
            b.append(Wl[1][:,np.newaxis])

        # Other layers doesn't contain any weights
        else:
            W.append([])
            b.append([])

    return (W,b,lname)
