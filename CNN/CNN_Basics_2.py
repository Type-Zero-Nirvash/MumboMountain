######################################################################
# In this example we will build upon what we discussed in
# the example: CNN_Basic_Model.py
#
# Here we will create multiple Models using different layers
# Explore and explain how we can increase the accuracy of our model
# Visualize our models performance and get some predictions

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from skimage import io
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

######################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--save",
                    help="Save graphs instead of showing them",
                    action="store_true")

args = parser.parse_args()

# Figure settings
for param in ['text.color',
              'axes.labelcolor',
              'xtick.color',
              'ytick.color',]:
    plt.rcParams[param] = 'navy'

plt.rcParams["figure.figsize"] = (20, 12)
######################################################################

######################################################################
# Sequential Models
#
# A Sequential Model can be used when our model consists of
# a plain stack of layers
# Where each layer has exactly ONE input/output tensor
#
# We should NOT use a Sequential Model if:
# Our model will need multiple inputs/outputs
#
# If any Layer within the model would use multiple inputs/outputs
#
# We need to do layer sharing
#
# We want a non-linear topology,
#    such as a residual connection or a multi-branch model
######################################################################

######################################################################
# Example Data
#
# For now we will use a 5 class dataset, of about 3600 images
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
dataset_dir = tf.keras.utils.get_file('flower_photos',
                                      origin=dataset_url,
                                      untar=True)

dataset_dir = pathlib.Path(dataset_dir)

######################################################################
# Lets define some parameters for this dataset

batch_size = 32
img_y = 180
img_x = 180

# Here we will be splitting our dataset so we can use a a set amount
# to VALIDATE the progress of our model

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.3, # we will use 30% of the data to validate
    subset="training",
    seed=123,
    image_size=(img_y, img_x),
    batch_size=batch_size
)

# validation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.3, # we will use 30% of the data to validate
    subset="validation",
    seed=123,
    image_size=(img_y, img_x),
    batch_size=batch_size
)

# Lets get a glipse of the data were working with
# class_names = train_ds.class_names
for images, labels in train_ds.take(1):
    for x in range(9):
        ax = plt.subplot(3, 3, x + 1)
        plt.imshow(images[x].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[x]], color='navy')
        plt.axis("off")

if args.save:
    plt.savefig("Example_Images.png")
else:
    plt.show()

######################################################################
# Here we are using buffered prefetching, to prevent I/O from
# blocking loading data from disk
#
# train_ds.cache()
# <dataset>.cache() -> this will keep our images loaded in memory
# after they are loaded on the first Epoch
# (instead of making us load from disk again)
#
# Additionally, if you are using a dataset that is too large to be
# loaded into your memory, this same function can be used to
# create a performant on-disk cache
#
# We can also <dataset>.prefetch to improve our performance a bit more
# (What we would expect to see is less latency, but this will
#  require additional memory to store the prefetched data)

######################################################################
# Here I want to mention that when we run our models on GPU
# while we do greatly benefit from the large amount of
# threads available
# We may face speed decreases when we are loading data
#
# We may discuss GPU usage in more detail in another example
# But if you're interested,
# Try seeing the time differences between batch
# sizes for training your model
#
# You may see a trend of better performance with higher batch sizes
######################################################################

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

######################################################################
# Standardization
#
# Rescaling
# Here we will standardize our image data using a Rescaling layer
# Having smaller input values is generally a good idea so lets
# have our data be from 0 to 1 in value
#
# Resizing
# If you need to do additional resizing on your data,
# you can also include a Resizing layer in your model
#
# resize_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
#
# reshape_layer = layers.experimental.preprocessing.Resizing(height,
#                                                            width)
#
# We can use this layer by using a map, but we will include
# this layer in our model for simplicity

######################################################################
# Creating our Model
#
# We will start with a model that is very similar to the model we
# created in our previous example
#
# For now lets try to provide 5 classifications
#
# For your first run, uncomment this section below

"""
num_classes = 5

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255,
                                                input_shape=(img_y, img_x, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes),
])

######################################################################
# It would be difficult to explain each of these arguments we are
# using in this example
# But here is a small bit of information if it helps your understanding
#
# The 'adam' optimizer is a "stochastic gradient descent" function
# (there is a little more to it than that but for now just stick with me)
#
# If you don't know what a Gradient Descent function is, just think of this
#
# If you place a ball on a downward slope we know that it will roll
# down to the bottom right?
#
# Then once the ball reaches a point where it is no longer going
# down the slope, lets say it stops.
#
# Well, thats what our Gradient Descent does!
# It picks a point in our function, and travels down the slope
# until it finds the place at the lowest point
#
# The Stochastic part, (meaning random), makes this iterative
# process much less computationally expensive
#
# This is done by Stochastically selections just one datapoint
# to perform the calculations needed for Gradient Descent
# Instead of using EVERY datapoint we have.
#
# Gradient Descent is helpful, but very slow on larger datasets
######################################################################
# Sparse Categorical Cross Entropy
#
# This means we will calculate our models loss as the
# cross-entropy loss between labels and our models predictions.
#
# What is important to us here is that we want our model to provide
# more than one class labels, and so we must use this argument.
######################################################################
# Metrics
#
# Simply enough, this just helps us visualize our models performance
######################################################################
# Optional Considerations
#
# If you want to know more about what a Gradient Descent
# function does and why it helps us,
#
# I encourage you to search online!
#
# Getting better at breaking down questions you find yourself asking,
# and how to find answers to those questions is a VALUABLE SKILL
#
# Unfortunately, given the uninteractive nature of these guides and
# examples, I cannot directly help you with developing this skill
#
# But I can certainly encourage you to think about this process of,
# asking questions and looking for answers, as a skill that you can
# improve on!
######################################################################

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

print(model.summary())
"""

######################################################################
# Training our Model & Visualizing performance

"""
EPOCHS = 13

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Lets see how our Model is doing
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Val Accuracy")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

if args.save:
    plt.savefig("Starting_Performance.png")
else:
    plt.show()
"""

######################################################################
# Overfitting
#
# You might have noticed on our graph that our validation
# accuracy seems to be stuck around ~65%
#
# Since we do not also see our training accuracy following this trend
#
# This looks like we might be suffering from Overfitting
# Which means our model wont perform well when exposed to new data
#
# Lets explore how to improve our model and change this unwanted
# result

######################################################################
# Data Augmentation
#
# Since we might not always have the luxury of just finding
# more data suitable to our goal
#
# We can take advantage of Data Augmentation to try and get around
# this
#
# Essentially, we are going to use what data we have,
# to try and generate some additional training data
#
# Our goal is to have data that is realistically similar to our
# existing data, but to have this new data also create some
# variance
#
# Since we are working with image data, we can perform some simple
# transformations and use those to better train our model
#
# Remember, data is possibly the most important factor in what
# results you will see when training your models!

# This will act as another layer in our model
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal',
                                                 input_shape=(img_y,
                                                              img_x,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

######################################################################
# Dropout
#
# This is another technique we will use to see if we can improve
# our training results
#
# Dropout is a form of regularization,
# It works by removing random selections (of a set number),
# of nodes in a layer for a single gradient step.
#
# As a very rough example, we can think of this process as
# "encouraging" our model not to jump to conclusions
#
# We want our model to avoid learning tiny details, or potential
# noise in our data
#
# For example, if every image we had of a shoe accidentally had
# a time stamp on the image.
# Well if we asked our model to tell us if an image was of a shoe
# and the model couldn't find find that time stamp,
# it might not be able to accurately classify in this situation
#
# While that might not the best example,
# I want to lead us to the thought of
# We want our model to become better at knowing general details
#
# (While not the point of this section, we should know not to use
# data with strange features that may have negative effects on our
# model)
#
# After you have ran the original implementation,
# uncomment the lines below and see how these changes have
# improved our model
num_classes = 5

improved_model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes),
])

# Now lets compile and retrain our model and see the effects
improved_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

print(improved_model.summary())

######################################################################
# Training
#
# Make sure to have commented out the section above before running
# this section.

# Lets try using more training iterations as well
EPOCHS = 18

history = improved_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# How do our results change now?
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Val Accuracy")
plt.legend(loc='lower right')
plt.title('Data Augmentation & Dropout Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.legend(loc='upper right')
plt.title('Data Augmentation & Dropout Accuracy Loss')

if args.save:
    plt.savefig("Improved_Performance.png")
else:
    plt.show()

# With our changes we should see a 10% increase in accuracy
# There are still other changes we could make to continue
# to get better results
#
# Checkout the results of your own changes to help your understanding
######################################################################
