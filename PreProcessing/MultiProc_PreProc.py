#####################################################################
# Images by Jonathan lajoie from Pexels

import concurrent.futures
import glob
import os

import numpy as np
from skimage import io
from skimage.util import img_as_ubyte

#####################################################################
# Gaussian function

# Univariate Normal Distribution
def dNorm(x, mu, std_dev):
    return 1 / (np.sqrt(2 * np.pi) * std_dev) * np.e ** (-np.power((x - mu) / std_dev, 2) / 2)

def gKern(size, sigma=1.):

    kern_one = np.linspace(-(size // 2), size //2, size)

    for i in range(size):
        kern_one[i] = dNorm(kern_one[i], 0, sigma)

    # .T -> transpose arrays
    kern_two = np.outer(kern_one.T, kern_one.T)
    kern_two *= 1.0 / kern_two.max()

    return kern_two

def gaussian(img, size):
    kern = gKern(size, sigma=np.sqrt(size))
    return convolution(img, kern, avg=True)

# Expects gray image, dimension dependant
def convolution(img, kern, avg=False):

    img_row, img_col = img.shape
    kern_row, kern_col = kern.shape

    output = np.zeros(img.shape)

    pad_y = int((kern_row - 1) / 2)
    pad_x = int((kern_col - 1) / 2)
    pad_img = np.zeros((img_row + (2 * pad_y), img_col + (2 * pad_x)))

    pad_img[pad_y:pad_img.shape[0] - pad_y, pad_x:pad_img.shape[1] - pad_x] = img

    for row in range(img_row):
        for col in range(img_col):
            output[row, col] = np.sum(kern * pad_img[row:row + kern_row, col:col + kern_row])

            if avg:
                output[row, col] /= kern.shape[0] * kern.shape[1]

    return output


#####################################################################
# Helper Function
#
# We will need this helper function to allow us to perform our
# smoothing (pre-processing) on multiple files concurrently

def gauss_preProcess(filename):

    # Results will be placed into directory called 'Processed'
    #    With name: (original_filename)_smoothed.jpeg

    output_path = 'Processed/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    original_filename, file_extension = os.path.splitext(filename)
    original_filename = original_filename.split('/')[-1]
    filepath = "Processed/" + f"{original_filename}_smoothed{file_extension}"

    img = io.imread(filename, as_gray=True)
    img = gaussian(img, 5)
    img = img_as_ubyte(img) # convert to uint8 to supress warnings
    img = io.imsave(filepath, img)
    return filepath

with concurrent.futures.ProcessPoolExecutor() as executor:

    # Get list of all images in images directory
    # Make sure files have jpeg extension (change this to jpg if needed)

    # Note:
    #   .png should work since we load our images as gray-scale
    #   But extra vector dimension due to alpha chanel may give you issues
    #   I have not tested this
    images = glob.glob("images/*.jpeg")

    for images, filepath in zip(images, executor.map(gauss_preProcess, images)):
        print(f"{filepath} has been preprocessed, saved as: {images}")

#####################################################################
# Testing
#
# To test our script, I have provided only 3 images but you are
# welcome to test using your own images and see the time we save
# compared to doing our preprocessing asynchronously
#
# To test: python3 MultiProc_PreProc.py

