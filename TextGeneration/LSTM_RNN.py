import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

with open("GlitzPitDialog.txt") as f:

    fullText = f.read()
    fullText = fullText.lower()

# Make list of punctuation and delimiters
punct = sorted(list(set(fullText)))
punct_int = dict((k, v) for v, k in enumerate(punct))
int_punct = dict((k, v) for k, v in enumerate(punct))
print(punct)

# Dataset summarization
total_chars = len(fullText)
distinct_chars = len(punct)
print("Total Chars: {0}\nDistinct Chars: {1}".format(total_chars, distinct_chars))

#####################################################################
# Data prep
# Training pattern size will be 100
#
seq_length = 100
train_data_x = []
train_data_y = []

for item in range(0, total_chars - seq_length, 1):
    seq_in = fullText[item:item + seq_length]
    seq_out = fullText[item + seq_length]

    train_data_x.append([punct_int[char] for char in seq_in])
    train_data_y.append(punct_int[seq_out])

num_patterns = len(train_data_x)
print("Total patterns: {}".format(num_patterns))

#####################################################################
# Shape, normalize, and encode training data

# Reshape X-axis data to form: [samples, time steps, features]
X = numpy.reshape(train_data_x, (num_patterns, seq_length, 1))

# Normalize
X = X / float(distinct_chars)

# Encode output variable
y = to_categorical(train_data_y)

#####################################################################
# LSTM Model
# Here we define the layers of our model

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#####################################################################
# Our model will not have test data, we are using our entire text
# to train our model

# Checkoutpoint
filepath = "model_weights/weights-{epoch:02d}--{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#####################################################################
# Model fitting
# Feel free to adjust these values and see how this affects your
# loss value
#
# Allow our model to train before proceeding to Text Generation

# Comment out the line below after you have trained your model
# model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

#####################################################################
# Text Generation using our LSTM
#
# Load model weights
# Please select the model which achieved the smallest loss value

filename = "model_weights/weights-20--1.8392.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Generate a random seed
start = numpy.random.randint(0, len(train_data_x) - 1)
pattern = train_data_x[start]

print("\n")
print("Seed:")
print("\"", ''.join([int_punct[val] for val in pattern]), "\"")
print("\n")

results = ""

def sample(preds, temp=0.5):
    # help function to sample an index from probability array
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temp
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)

# Generate characters
for i in range(1000):

    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = (x / float(distinct_chars)) + (numpy.random.rand(1, len(pattern), 1) * 0.01)

    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_punct[index]

    seq_in = [int_punct[val] for val in pattern]
    results += result

    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print(results)

# Critique our results
#####################################################################

