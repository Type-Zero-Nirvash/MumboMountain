#####################################################################
# Activation Functions
#
# Without an Activation Function, a neural network is just a
# Linear Regression model. Activation Functions calculate
# NON-Linear transformations on input, which creates greater
# flexibility in neuron activation.
#

#####################################################################
# Consider a Linear function
# f = cx
#
# lookup gradient descent
#
# What effect would this function have when used in a model with
# multiple layers?
#
# If we consider Gradient Descent, since we know the derivative of
# our linear function will result in a constant gradient,
# then we should be aware not to create sequential linear layers
# in our model, as the result would not differ from having
# a single linear layer.
#
# Gradient Descent:
#  an algorithm that estimates where a given function will produce
#  the lowest output values.

#####################################################################
# Example Activation Functions
#
#####################################################################
# Sigmoid: 1 / (1 + e ^ (-x))
# Results between 0 and 1
#
# Pros: makes clear distinctions on predictions, output range is small
#
# Cons: For both ends of the function Y values change much less
#       in respect to X, called vanishing gradient problem
#
#####################################################################
# tanh: tanh(x) = 2 / (1 + e ^ (-2x)) OR 2 sigmoid(2x) -1
# Results between -1 and 1
#
# Pros: same pros as Sigmoid but with stronger gradient than Sigmoid
#
# Cons: Vanishing gradient
#
#####################################################################
# ReLU: max(0, x)
# Results 0 or greater
#
# Pros: When used instead of Sigmoid or tanh, activation is not as
#       dense, which should allow for more efficient models,
#       computationally cheaper than Sigmoid or tanh
#
# Cons:
# Because of the horizontal line (when x is negative),
# our gradient can trend towards 0.
#
# For activations in that region of ReLU, the gradient will be 0
# since the weights will no longer be adjusted during descent.
#
# When this occurs the neurons involved will no longer change
# in response to variations in input.
# (Since 0 will always be our lower bound limit)
#
# This is called the dying ReLU problem.
#
# Variations of ReLU attempt to account for this downfall.
#
#####################################################################
# Leaky ReLU: max(0.1x, x)
#
# More Activation Functions are available such as Maxout, ELU, etc

#####################################################################
# Examples
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--save",
                    help="Save graphs instead of showing them",
                    action="store_true")

args = parser.parse_args()

def sigmoid(x):

    z = 1 / (1 + np.exp(-x))
    dz = z * (1 - z)
    return z, dz

def tanh(x):

    z = ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))
    dz = 1. - (z ** 2.)
    return z, dz

def ReLU(x):
    return np.maximum(0, x)

def Leaky_ReLU(x):

    return np.maximum(x * 0.01, x)

def softmax(x):

    ex = np.exp(x - np.max(x))
    return ex / ex.sum(axis=0)

#####################################################################

plt.style.use('dark_background')

colors = [
    '#F7B538', # Yellow / Red
    '#DB7C26', # Light Orange
    '#D8572A', # Orange
    '#C32F27', # Red
    '#780116', # Maroon
]

for param in ['text.color',
              'axes.labelcolor',
              'xtick.color',
              'ytick.color',]:
    plt.rcParams[param] = '0.9'

plt.rcParams['axes.facecolor'] = "#2A3459"

x = np.arange(-6, 6, 0.01)
z = np.arange(-4, 4, 0.01)

# Sigmoid
fig, ax = plt.subplots(figsize=(16, 9))

# Grid settings
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.plot (x, sigmoid(x)[0], color=colors[0], linewidth=4, label="sigmoid")
ax.plot (x, sigmoid(x)[1], color=colors[1], linewidth=4, label='dX')
ax.legend(loc='upper left', frameon=False,)

if args.save:
    plt.savefig("Sigmoid.png")
else:
    plt.show()

# Tanh
fig, ax = plt.subplots(figsize=(16, 9))

# Grid settings
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.plot (z, tanh(z)[0], color=colors[2], linewidth=4, label="tanh")
ax.plot (z, tanh(z)[1], color=colors[3], linewidth=4, label='dZ')
ax.legend(loc='upper left', frameon=False)

if args.save:
    plt.savefig("Tanh.png")
else:
    plt.show()

# ReLU
fig, ax = plt.subplots(figsize=(16, 9))

# Grid settings
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.plot (x, color=colors[4], linewidth=4, label="ReLU")
ax.plot (ReLU(x), color=colors[0], linewidth=4, label='dZ')
ax.legend(loc='upper left', frameon=False)

if args.save:
    plt.savefig("ReLU.png")
else:
    plt.show()
