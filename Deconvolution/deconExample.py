#####################################################################
# Deconvolution examples for signals and images
# Image by Jonathan lajoie from Pexels

#####################################################################
# Deconvolution
#
# Used to improve contrast and resolution on digital images
# Useful for microscope imagery
#
# Very beneficial to have three-dimensional collection of images
# of biological sample
#
# Think of how a 3D object is printed, often we can see layers
# being stacked to create the overall shape of the object
#
# Given these different optical cross sections we can Deconvolve
# Biological sample imagery to obtain a higher quality visual
# of the sample
#
# As we shift along the Z-axis, considering each image of our
# sample set.
# Certain features will change in focus, and contribute to
# more definite edges of these features


#####################################################################
# Image Quality
#
# Image quality can be affected by four sources
# Noise - in digital microscopy, noise is mainly sourced from
#         the signal itself, or the digital imaging system.
#
# Noise is not generated entirely randomly, but can be seen in
#   imagery as what we may think of as:
#      grain, discoloration (salt & pepper), etc
#
#
# Scatter - random light disturbances caused by refraction
#           within specimen.
#           Scatter is considered truly random.
#
# Glare - random light disturbance, caused by light passage
#         through optical equipment elements.
#         Consider light passage through glass elements
#         these could produce glare.
#
# Blur - non-random light spread, caused by refraction of
#        light passage through imaging system.
#        Diffraction is the largest source of blur.
#####################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import deconvolve

plt.style.use('dark_background')

#####################################################################
# Basic example to show deconvolution effect on simple signal

# Square signal generation
sig = np.repeat([0.0, 1.0, 0.0], 100)

# Gaussian smoothing
gaussian = np.exp( -((np.linspace(0, 50) -25.0) / float(12)) ** 2 )

#####################################################################
# Convolution
filtered_sig = np.convolve(sig, gaussian, mode='same')

#####################################################################
# Deconvolution
decon, _ = deconvolve(filtered_sig, gaussian)
n = len(sig) - len(gaussian) + 1
s = (len(sig) - n) / 2

# Deconvolve from both directions ?
decon_results = np.zeros(len(sig))
decon_results[int(s):len(sig) - int(s) - 1] = decon
decon = decon_results

#####################################################################
# Our results should be very similar to our original signal

fix, axes = plt.subplots(nrows=4, figsize=(16,9))

axes[0].plot(sig, color='pink', label='Generated signal', lw=2)
axes[1].plot(gaussian, color='yellow', label='Gaussian Filter (to be applied)', lw=2)

# Normalize convolution (divide by sum of filter window)
axes[2].plot(filtered_sig / np.sum(gaussian), label="Convoluted", color='skyblue', lw=2)
axes[3].plot(decon, color='red', label="Deconvoluted result", lw=2)

for i in range(len(axes)):
    axes[i].set_xlim([0, len(sig)])
    axes[i].set_ylim([-0.075, 1.25])
    axes[i].legend(loc=1, fontsize=12)

    if not i == len(axes) - 1:
        axes[i].set_xticklabels([])

plt.show()

#####################################################################
# Image Example using scikit-image and scipy
# Using Richardson - Lucy

from scipy.signal import convolve2d as conv2
from skimage import color, io, restoration

try:
    img = io.imread("Gorilla.jpeg")
    img = color.rgb2gray(img)
except FileNotFoundError:
    print("No image file found")

rng = np.random.default_rng()

psf = np.ones((5, 5)) / 25
img = conv2(img, psf, mode='same')

# Generate noise on image
noisy_img = img.copy()
noisy_img += (rng.poisson(lam=25, size=img.shape) - 10) / 255.

# Deconvolve using Richardson-Lucy
decon_img = restoration.richardson_lucy(noisy_img, psf, iterations=33)

# Plot results
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
plt.gray()

axes[0].imshow(img)
axes[0].set_title('Original')

axes[1].imshow(noisy_img)
axes[1].set_title('Noise added')

axes[2].imshow(decon_img, vmin=noisy_img.min(), vmax=noisy_img.max())
axes[2].set_title('Richardson Lucy')

axes[0].axis('off')
axes[1].axis('off')
axes[2].axis('off')

fig.subplots_adjust(wspace=0.05, hspace=0, top=1, bottom=0, left=0, right=1)

plt.show()

#####################################################################
# Image example with manual Gaussian function
# Using unsupervised wiener

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
# Testing

try:
    img = io.imread("Gorilla.jpeg")
    img = color.rgb2gray(img)
except FileNotFoundError:
    print("No image file found")

# Gaussian filter as shown above
img_blur = gaussian(img, 5)

decon_img, _= restoration.unsupervised_wiener(img_blur, psf)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
plt.gray()

axes[0].imshow(img)
axes[0].set_title('Original')

axes[1].imshow(img_blur)
axes[1].set_title('Blurred image')

axes[2].imshow(decon_img, vmin=img_blur.min(), vmax=img_blur.max())
axes[2].set_title('Unsupervised Wiener')

axes[0].axis('off')
axes[1].axis('off')
axes[2].axis('off')

fig.subplots_adjust(wspace=0.05, hspace=0, top=1, bottom=0, left=0, right=1)
plt.show()

#####################################################################

