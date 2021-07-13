import os.path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFilter
from skimage import filters
from skimage import feature
from skimage import util
from skimage.data import camera
from skimage.util import compare_images
from skimage import io

#####################################################################
# Used for histogram graphs
def getRed(R): return '#%02x%02x%02x' % (R, 0, 0)
def getGreen(G): return '#%02x%02x%02x' % (0, G, 0)
def getBlue(B): return '#%02x%02x%02x' % (0, 0, B)

#####################################################################
# Resulting image file names
iGray = 'Gray.png'
iBlack = 'BlackWhite.png'
iHist = 'Histogram.png'
iBlur = 'Blur.png'
iSobel = "Sobel Edge.png"
iCan = "Canny Edge.png"
iNoise = "SPNoise.png"

#####################################################################
# Open Test Image

try:
    img = Image.open("Rose.jpg")
except FileNotFoundError:
    print("Image File not found...")

print(
    "Format: {0}\nHeight/Width: {1}\nColor Type: {2}".format(img.format, img.size, img.mode))

#####################################################################
# Grayscale Image
if not os.path.isfile(iGray):
    imgGray = img.convert("1")
    imgGray.save(iGray)

#####################################################################
# Binary Image
if not os.path.isfile(iBlack):
    threshold = 200
    def fn(x): return 255 if x > threshold else 0
    blackWhite = img.convert('L').point(fn, mode='1')
    blackWhite.save('BlackWhite.jpg')

#####################################################################
# RGB Histogram
if not os.path.isfile(iHist):
    hst = img.histogram()
    Red = hst[0:256]  # indicates Red
    Green = hst[256:512]  # indicated Green
    Blue = hst[512:768]  # indicates Blue

    fig, (axR, axG, axB) = plt.subplots(1, 3, figsize=(12, 8))

    fig.suptitle("RGB Histogram")
    axR.set_title("Red")
    axG.set_title("Green")
    axB.set_title("Blue")

    for i in range(0, 256):
        axR.bar(i, Red[i], color=getRed(i), alpha=0.3)

    for i in range(0, 256):
        axG.bar(i, Green[i], color=getGreen(i), alpha=0.3)

    for i in range(0, 256):
        axB.bar(i, Blue[i], color=getBlue(i), alpha=0.3)

    fig.tight_layout()
    plt.savefig(iHist)

#####################################################################
# Gaussian Blur
if not os.path.isfile(iBlur):
    fig, axes = plt.subplots(1, 4, figsize=(12, 8))
    ax = axes
    fig.suptitle("Gaussian Blur")

    BlurOne = img.filter(ImageFilter.GaussianBlur(5))
    BlurTwo = img.filter(ImageFilter.GaussianBlur(15))
    BlurThree = img.filter(ImageFilter.GaussianBlur(45))

    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].tick_params(left=False,
                      bottom=False,
                      labelleft=False,
                      labelbottom=False)

    ax[1].imshow(BlurOne)
    ax[1].set_title("Radius 5")
    ax[1].tick_params(left=False,
                      bottom=False,
                      labelleft=False,
                      labelbottom=False)

    ax[2].imshow(BlurTwo)
    ax[2].set_title("Radius 15")
    ax[2].tick_params(left=False,
                      bottom=False,
                      labelleft=False,
                      labelbottom=False)

    ax[3].imshow(BlurThree)
    ax[3].set_title("Radius 45")
    ax[3].tick_params(left=False,
                      bottom=False,
                      labelleft=False,
                      labelbottom=False)

    fig.tight_layout()
    plt.savefig(iBlur)

imgCam = io.imread("Rose.jpg", as_gray=True)

#####################################################################
# Edge detection
if not os.path.isfile(iSobel):

    edge_sobel = filters.sobel(imgCam)

    fig, axes = plt.subplots(ncols=1, sharex=True, sharey=True,
                             figsize=(12, 8))

    axes.imshow(edge_sobel, cmap=plt.cm.gray)
    axes.set_title('Sobel Edge Detection')

    plt.tight_layout()
    plt.savefig("Sobel Edge.png")

#####################################################################
# Canny Edge
if not os.path.isfile(iCan):

    edges = feature.canny(
        image=imgCam,
        sigma=2.0,
        low_threshold=0.1,
        high_threshold=0.3,
    )

    fig, axes = plt.subplots(ncols=1, sharex=True, sharey=True,
                             figsize=(12, 8))

    axes.imshow(edges, cmap=plt.cm.gray)
    axes.set_title('Canny Edge Detection')

    plt.tight_layout()
    plt.savefig(iCan)

#####################################################################
# Noise Gen
if not os.path.isfile(iNoise):

    imgCam = io.imread("Rose.jpg")

    noisy = util.random_noise(imgCam, mode="s&p", amount=0.4)

    fig, axes = plt.subplots(ncols=1, sharex=True, sharey=True,
                             figsize=(12, 8))

    axes.imshow(noisy)
    axes.set_title('Salt & Pepper Noise')

    plt.tight_layout()
    plt.savefig(iNoise)

#####################################################################
# Calculate gradient vector
with open("ImageVectors.txt", "w") as resFile:

    imgArray = np.asarray(img)

    resFile.write("Image Original vector: \n\n{}\n\n".format(imgArray))
    resFile.write("Image Gradient vector: \n\n{}\n\n".format(
        np.gradient(imgArray)))

