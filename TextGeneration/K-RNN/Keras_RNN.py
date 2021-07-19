#####################################################################
# Text Generation - Model creation and training
# Using Tensorflow / Keras
#
# All training text from "SpongeBob Squarepants"
#####################################################################

import os, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

#####################################################################
# Read Data

with open("Sponge.txt") as f:
    fullText = f.read()
    fullText = fullText.lower()

unique = sorted(set(fullText))

#####################################################################
# Text Vectorization
#
# Our text needs to be in a numerical format to be used with our model
#
# TensorFlows StringLookup will convert each character into a number
# but we must first tokenize our text
#
# Try using unique chars for this

chars = tf.strings.unicode_split(unique, input_encoding='UTF-8')
print("\n" * 10)
print(chars)

# Now we can create our StringLookup Layer
char_to_id = preprocessing.StringLookup(vocabulary=list(unique), mask_token=None)

# We will also want to read our results,
# so let us create an additional Layer
#
# We use get_vocabulary() so that the [UNK] tokens are set the same
# finally, invert=True will allow us to convert our results
id_to_char = preprocessing.StringLookup(vocabulary=char_to_id.get_vocabulary(), invert=True, mask_token=None)

# Note: these are character based tokens, not tokenized words themselves
numeric_tokens = char_to_id(chars)

# Use this function to convert our results
def make_readable(ids_to_convert):
    return tf.strings.reduce_join(id_to_char(ids_to_convert), axis= -1)

#####################################################################
# Reoccuring Neural Networks
#
# RNN's are good at tasks that involve sequential data.
#
# When we think of language, it's not too strange to say it's
# a sequence of characters that make up words, or a sequence
# of words that make a sentence, or a sequence of sentences that
# make a paragraph, etc.
#
# Standard Neural Networks, such as Convolutional, do not work in
# this way.
# They are not designed to give consideration to their previous
# inputs in the way that a RNN is.

#####################################################################
# Problem/Goal analysis
#
# When we consider our goal of generating text that can best
# resemble intelligable language. We want to take advantage
# of the sequential nature of the data, and since our RNN will
# generate output with consideration of previous states, this
# should work well for us.

#####################################################################
# Batching
#
# Here we will want to break up our data so that we can fit our
# model to our training data. As well as have testing data to
# ask our model to generate predictions for when given.

# This is converting our text into a stream of character indices

all_ids = char_to_id(tf.strings.unicode_split(fullText, 'UTF-8'))
id_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

#for ids in id_dataset.take(10):
#    print(id_to_char(ids).numpy().decode('utf-8'))

seq_len = 100
words_per_epoch = len(fullText) // (seq_len + 1)

sequences = id_dataset.batch(seq_len + 1, drop_remainder=True)

for seq in sequences.take(5):
    print(make_readable(seq).numpy())

def split_input(sequence):
    in_text = sequence[:-1]
    target_text = sequence[1:]
    return in_text, target_text

dataset = sequences.map(split_input)

# Batch size - consider the size of your dataset
BATCH_SIZE = 37

# Buffer size used for input shuffling
BUFFER_SIZE = 333

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

#####################################################################
# Building our Model
#
# We will use 3 layers
#
# Embedding:
# input layer, a trainable lookup table for mapping character ids
# to a vector with 'embedding_dim' dimensions
#
# GRU: (This can be replaced with an LSTM layer)
# RNN with size 'units = rnn_units'
#
# Dense:
# output layer, with vocab_size outputs.
# Outputs one logit for each character in a vocabulary.

# Length of our 'vocabulary', unique characters
vocab_size = len(unique)

# Embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

class TextModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)

        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim)

        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)

        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False,
             training=False):

        x = inputs
        x = self.embedding(x, training=training)

        if states is None:
            states = self.gru.get_initial_state(x)

        x, states = self.gru(x, initial_state=states,
                             training=training)

        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

#####################################################################

model = TextModel(
    vocab_size=len(char_to_id.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
)

#####################################################################
# What is going on?
#
# For each item in our vocabulary,
# our model will check our lookup table - Embedding layer
# run the GRU one timestep, with the embedding as input
# and applies the dense layer to generate logits
# predicitng what is likely to be the next character

#####################################################################
# Model Check

print("\n" * 10)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape,
          '# (batch_size, sequence_length, vocab_size)')

model.summary()

#####################################################################
#
# Note:
#    remember to 'sample' from the distribution
#    if we take the argmax of the distribution we may loop infinitely

sampled_indices = tf.random.categorical(example_batch_predictions[0],
                                        num_samples=1)

sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

#####################################################################
# What do you expect?
#
# If we ask our untrained model for predictions, what will we see?

print("\n" * 10)

print("Input:\n", id_to_char(input_example_batch[0]).numpy())
print("\nNext Char Predictions:\n", id_to_char(sampled_indices).numpy())

print("\n" * 10)

#####################################################################
# Before Training
#
# Since we have not yet trained our model,
# these first results should reflect that.
#
# Before training, the exponent of our mean loss should equal
# our vocab_size
#
# Higher loss here would mean our model is confident of its wrong
# answers, and there is an underlying issue

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()

print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, seq_length, vocab_size)")
print("Mean loss:        ", mean_loss)

print("Vocab size:            ", vocab_size)
print("Exponential Mean Loss: ", tf.exp(mean_loss).numpy())

# Configure training procedure
model.compile(optimizer='adam', loss=loss)

# Directory where we will save checkpoints
checkpoint_dir = './train_checkpoints'

# Checkpoint file names
checkpoint_name = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_name,
    save_weights_only = True
)

#####################################################################
# Training Method 1
#
# Once you have trained your model, remember to comment out the
# model fitting line below if you do not wish to wait for retraining

# Try training with different EPOCHS
# (try a low number the first time to see how your model changes)

# This section has been commented out since we will use a
# different training sequence at the end of the example
#
# You may want to uncomment the following section
# and comment out the section:
# Training Method 2

"""
EPOCHS = 25

history = model.fit(
    dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback])

"""

#####################################################################
# Single Step Prediction
#
# Each time we call our model we pass in text and an internal state
#
# Then, the model returns its prediction for the next character
# and its new state
#
# If we continue to pass in the prediction and state,
# the model will continue generating text.
#
# Example:
# (Assuming our model has been trained well)
#
# Say our training text contained the word 'apple' frequently,
# now when we give our model the character 'a'
# the model may produce 'p' in response.
#
# Now the model will remember this new state,
# then if we give the model the character 'p'
# the model may then produce 'l'
#
# This single step pattern of prediction is how our model works!
#
# Consider that our model does not have any concept of:
# words, or grammar, or sentence structure, etc
#
# Yet our model is (potentially) capable of generating text
# that seems coherent.
#
# Generated character by character
#####################################################################

#####################################################################
# We can use this class below to generate predictions from our model

class SingleStep(tf.keras.Model):

    def __init__(self, model, char_to_id, id_to_char, temperature=0.34):
        super().__init__()

        self.temperature = temperature
        self.model = model
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char

        # Mask used to prevent '[UNK]' from being generated
        skip_ids = self.char_to_id(['UNK'])[:, None]

        sparse_mask = tf.SparseTensor(

            # Puts a -inf at each unwanted index
            values = [-float('inf')] * len(skip_ids),
            indices = skip_ids,

            # Match shape to the vocabulary
            dense_shape = [len(char_to_id.get_vocabulary())]
        )

        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_single_step(self, inputs, states=None):

        # Convert words to token IDs
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.char_to_id(input_chars).to_tensor()

        # Run our model
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids,
                                              states=states,
                                              return_state=True)

        # Only use last prediction
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature

        # Apply our mask to prevent [UNK] from generating
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs
        predicted_ids = tf.random.categorical(predicted_logits,
                                              num_samples=1)

        predicted_ids = tf.squeeze(predicted_ids, axis = -1)

        # Convert from ids to characters
        predicted_chars = self.id_to_char(predicted_ids)

        # Return our prediction and current model state
        return predicted_chars, states

#####################################################################
# Training Method 1 Results
#
# Uncomment the following section if you are following
# Training Method 1
#
# Try a different number for these variables named above:
# EPOCHS
# BATCH
# and BUFFER_SIZE
#
# Then see how this affects the results your model produces

"""
single_step_model = SingleStep(model, char_to_id, id_to_char)

start = time.time()
states = None

next_char = tf.constant(['SpongeBob:'])
result = [next_char]

for n in range(1000):
    next_char, states = single_step_model.generate_single_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()

print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
print('\nRun time: ', end - start)
"""

#####################################################################
# Saving our model
#
# Once you are getting results you are satisfied with,
# you will likely want to save your model
#

# tf.saved_model.save(single_step_model, 'single_step')

# To load it for future use, we can do it as so
# single_step_model = tf.saved_model.load('single_step')

#####################################################################
# Moving forward
#
# The training procedure we used for this example was simple
# for the sake of understanding
#
# Our model was trained using 'teacher-forcing'
# This type of training does not prevent bad predictions
# from being fed back into the model
# Because of this, once our model forms a bad habit
# it will continue to make the same mistakes
#
# Now we will improve our training sequence
#####################################################################

#####################################################################
# Use the name of the model you created at the beginning of our
# example

class CustomTraining(TextModel):

    @tf.function
    def train_step(self, inputs):

        inputs, labels = inputs

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return {'loss' : loss}

model = CustomTraining(
    vocab_size = len(id_to_char.get_vocabulary()),
    embedding_dim = embedding_dim,
    rnn_units = rnn_units
)

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True))

#####################################################################
# Training Method 2
# Remember to comment out Training Method 1 before running the
# method below

EPOCHS = 20

mean = tf.metrics.Mean()

for epoch in range(EPOCHS):

    start = time.time()
    mean.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        logs = model.train_step([inp, target])

        mean.update_state(logs['loss'])

        if batch_n % 50 == 0:
            template = f"Epoch {epoch+1} Batch {batch_n} Loss {logs['loss']:.4f}"
            print(template)


    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_name.format(epoch=epoch))


    print()
    print(f'Epoch {epoch+1} Loss: {mean.result().numpy():.4f}')
    print(f'Time taken for 1 epoch {time.time() - start:.2f} sec')
    print("_" * 80)

model.save("Trained_Model")

#####################################################################
# Optional
# Load a saved model

#model.load_weights('./train_checkpoints/ckpt_9.index')

#####################################################################
# Training Method 2 Results

single_step_model = SingleStep(model, char_to_id, id_to_char)

start = time.time()
states = None

next_char = tf.constant(['SpongeBob:'])
result = [next_char]

for n in range(1000):
    next_char, states = single_step_model.generate_single_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()

print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
print('\nRun time: ', end - start)

#####################################################################

#####################################################################
# Play with your results and consider what effects your changes
# are creating
#####################################################################
