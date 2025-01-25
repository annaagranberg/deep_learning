import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import convolve
from skimage import color

def evaluate(model, dataset, final=False):
    print('Model performance:')
    
    # Evaluate on the training set
    train_score = model.evaluate(dataset.x_train_gray, dataset.x_train, verbose=False)
    print('\tTrain loss:          %0.4f' % train_score[0])
    
    # Validation or test evaluation
    if final:
        test_score = model.evaluate(dataset.x_test_gray, dataset.x_test, verbose=False)
        print('\tTest loss:           %0.4f' % test_score[0])
    else:
        val_score = model.evaluate(dataset.x_valid_gray, dataset.x_valid, verbose=False)
        print('\tValidation loss:     %0.4f' % val_score[0])
    
    return train_score

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

# Compute SSIM between two images
def compute_ssim(image1, image2):
    """Compute Structural Similarity Index (SSIM) between two images."""
    # Specify win_size explicitly to avoid errors
    ssim_score = ssim(image1, image2, data_range=image1.max() - image1.min(), channel_axis=2, win_size=11)
    return ssim_score

def compute_snr(image1, image2):
    """
    Compute the Signal-to-Noise Ratio (SNR) between two images.
    SNR = 10 * log10 (signal power / noise power)
    """
    # Ensure images are of the same scale (e.g., normalize to 0–1)
    image1 = image1 / image1.max()
    image2 = image2 / image2.max()
    
    signal_power = np.mean(image1 ** 2)
    noise_power = np.mean((image1 - image2) ** 2)
    
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))  # Add epsilon for stability
    return snr

# SCIELAB metric
def compute_scielab(image1, image2):
    """
    Compute SCIELAB perceptual color difference between two images.
    Converts RGB images to CIELAB space and calculates the difference.
    """
    # Convert RGB to CIELAB color space
    lab1 = color.rgb2lab(image1)
    lab2 = color.rgb2lab(image2)
    
    # Compute perceptual difference
    deltaE = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))
    
    # Apply a Gaussian contrast sensitivity filter (CSF)
    csf = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16  # 3x3 Gaussian filter
    deltaE_filtered = convolve(deltaE, csf, mode='reflect')
    
    # Calculate the mean difference as SCIELAB score
    scielab_score = np.mean(deltaE_filtered)
    return scielab_score