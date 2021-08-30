######################################################################
# Transfer Learning
# Comparing VGG16 to our previous CNN
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

######################################################################
# Setup

batch_size = 32
num_classes = 3

img_x = 224
img_y = 224

train_data = "Class_Data/seg_train"
test_data = "Class_Data/seg_test"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(img_y, img_x),
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_y, img_x),
    batch_size=batch_size,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_y, img_x),
    batch_size=batch_size,
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

base_model = VGG16(weights="imagenet",
                   include_top=False,
                   input_shape=(img_y, img_x, 3))

base_model.trainable = False

print(base_model.summary())

# now we add the layers we need to fit our goal

flat_layer = layers.Flatten()
dense_one = layers.Dense(256, activation='relu')
dense_two = layers.Dense(128, activation='relu')
out_layer = layers.Dense(6, activation='softmax')

transfer_model = Sequential([
    base_model,
    flat_layer,
    dense_one,
    dense_two,
    out_layer,
])

transfer_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

print(transfer_model.summary())

early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                         mode='max',
                                         patience=5,
                                         restore_best_weights=True)

transfer_model.fit(train_ds,
                   epochs=15,
                   validation_data=val_ds,
                   callbacks=[early])

transfer_model.save('VGG16_transfer')


######################################################################
# Our Model
#
# If you wish to run this example
# Comment out the section above right above the lines
# importing VGG16 packages
#
# To right above this comment block
"""

early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                         mode='max',
                                         patience=5,
                                         restore_best_weights=True)

our_model = Sequential([
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
    layers.Dense(6),
])

# Now lets compile and retrain our model and see the effects
our_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

print(our_model.summary())

history = our_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early],
)

our_model.save('Our_Model')
"""
