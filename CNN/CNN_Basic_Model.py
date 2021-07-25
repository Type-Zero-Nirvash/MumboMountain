#####################################################################
# Convolutional Neural Network
#
# In this example we will create a CNN that can classify
# multiple classes of images
#
# We will also aim to build upon on what we may already know
# about Convolution

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--save",
                    help="Save graphs instead of showing them",
                    action="store_true")

args = parser.parse_args()
######################################################################

#####################################################################
# Download our example dataset

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize image pixel values to be between 0 ~ 1
train_images, test_images = train_images / 255., test_images / 255.

# These will be the classications our model will make
class_names = ['airplane',
               'automobile',
               'bird',
               'cat',
               'deer',
               'dog',
               'frog',
               'horse',
               'ship',
               'truck']

# Lets see a sample set from the data
# (They will be blurry, this is not an error)
plt.figure(figsize=(12, 12))

for img in range(25):
    plt.subplot(5, 5, img + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[img])
    plt.xlabel(class_names[train_labels[img][0]])

if args.save:
    plt.savefig("Dataset_Sample.png")
else:
    plt.show()

#####################################################################
# Layers we will use:
#
# Conv2D layer:
#
# Creates a convolution kernel that is covolved with the layers
# input and produces a tensor of outputs.
#
# When we use this layer as the first layer in a model,
# we must use the argument 'input_shape'
# which expects a tuple of integers, or None
# -> (tuple - a finite ordered list)
#    [does not mean ordered by magnitude, just that there is an order]
######################################################################
# MaxPool2D:
#
# Downsamples the input using its dimensionality (height & width)
# by taking the MAX value over an input window (think batch)
# for each channel of the input
# The window is shifted by the 'strides' argument, if used
#
# The output, when using the 'valid' padding argument,
# has a spatial shape (rows and columns) of:
#
# out_shape = floor((in_shape - pool_size) / strides)
#             + 1 "When" in_shape >= pool_size
#
# When using the 'same' argument:
# out_shape = floor((in_shape - 1) / strides) + 1
#####################################################################
# Considerations:
#
# Recall previous examples in which we made our own
# Convolution of an image with a Gaussian function?
# (This is also known as a Weierstrass transform)
#
# Also recall our example on Activation Functions,
# we will use ReLU as our activation function for each layer
# in this model
# (ReLU -> Rectified Linear Unit)
#
# Remember that activation functions are used to create
# NON-Liner transformations on our data
# Since multiple layers of linear functions result in no change
# in layer output!
#
# You are welcome to use different activation functions and
# see how this affects your results
#####################################################################

#####################################################################
# Model creation

model = models.Sequential()
model.add(layers.Conv2D(32,
                        (3, 3),
                        activation='relu',
                        input_shape=(32, 32, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64,
                        (3, 3),
                        activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64,
                        (3, 3),
                        activation='relu'))

#####################################################################
# Model summary
model.summary()

# When looking at the model summary,
# note that the output of every layer is a 3D tensor representing:
# (height, width, channels)
#
# The height and width dimensions tend to shrink as we move through
# layers
#
# The number of output channels for each Conv2D are set by
# the first argument (32 or 64)
#
# We mention that our dimensions tend to shrink so that you may
# consider that when this occurs,
# it becomes computationally cheaper to add more
# output channels in each Conv2D layer

#####################################################################
# Getting our classification results
#
# To finish our model,
# we need to direct the output from the last layer (Conv2D)
#   (Note the shape: (4, 4, 64))
# Into one or more Dense layers
#
# Dense layers take vectors as their input (1D), not Matricies
#
# Since our output is in the form of a 3D tensor,
# We need to flatten, or unroll, our 3D output into a useable 1D
#
# Then we will use more Dense layers for generating our classifcation
#
# To generate 10 classifications, our final Dense layer
# will need to have 10 outputs
#####################################################################

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# If we check our model summary again,
# we will see that our previous final output of (4, 4, 64)
# is now flattened into a vector of shape 1024
# then processed through our two Dense layers to provide 10 classes

#####################################################################
# Compile and Train our Model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images,
                    train_labels,
                    epochs=10,
                    validation_data=(test_images, test_labels))

# Plotting our accuracy
plt.plot(history.history['accuracy'], label='accuracy', color='pink')
plt.plot(history.history['val_accuracy'], label='val_accuracy', color='skyblue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

if args.save:
    plt.savefig("Model_Accuracy.png")
else:
    plt.show()

test_loss, test_acu = model.evaluate(test_images,
                                     test_labels, verbose=2)

# In this short example, we do not expect superb accuracy
# But, if you recall some of the methods we have introduced
# in previous examples.
#
# You are free to try making changes and seeing the effects
# on our model
print(test_acu)

#####################################################################
