import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

######################################################################
# Set graph styling
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
plt.style.use('dark_background')

# Read training data
training_data = pd.read_csv('train_assignment.csv')

# Display statistics for SalePrice
print(training_data['SalePrice'].describe())

# Plot of SalePrice
sns.displot(training_data['SalePrice'])
plt.show()

# Feature Heatmap of SalePrice correlation
corrmat = training_data.corr()
k = 9
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
f, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(training_data[cols].corr(), vmax=0.8, square=True, annot=True)
plt.show()

######################################################################
# Linear Regression
# Feature selection
x = training_data['GrLivArea'].values[:, np.newaxis]
y = training_data['SalePrice'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = LinearRegression().fit(x_train, y_train)

r_sq = model.score(x,y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

# Results for training data
y_pred = model.predict(x_test)

plt.rcParams["figure.figsize"] = (16, 9)
plt.scatter(x_test, y_test, color='pink', marker='^', alpha=0.5)
plt.plot(x_test, y_pred, color='blue')

plt.title("Living Area vs Price with prediction")
plt.xlabel("Sq Ft")
plt.ylabel("Price")
plt.show()

#####################################################################
# Working with Test data

test_data = pd.read_csv('test_assignment.csv')

a = test_data['GrLivArea'].values[:, np.newaxis]
b = test_data['LotArea'].values[:, np.newaxis]
c = test_data['FirstFlrSF'].values[:, np.newaxis]
d = test_data['Fireplaces'].values[:, np.newaxis]

# Predict price based on Test data
test_pred = model.predict(a)

plt.rcParams["figure.figsize"] = (16, 9)
plt.scatter(x_test, y_test, color='red', marker='^', alpha=0.5)
plt.plot(a, test_pred, color='green')
plt.title("Test Data Prediction")
plt.xlabel("Sq Ft")
plt.ylabel("Price")
plt.show()

#####################################################################
# Manual Linear Regression
#
# Starting with the equation for a line we have:
#    y = mX + b
#
# Where:
#  Y is our predicted value
#  m is our coefficient
#  X is our predictor
#  b is our intercept
#

#####################################################################
# Mean Squared Error Cost Function (MSE)
#
# MSE measures how much the average model predictions vary
# from the correct values
#
# The higher our MSE the worse our model performs


#####################################################################
# One Half Mean Squared Error (OHMSE)
#
# A modification to MSE - we multiply by (1/2) so when we take
# the derivative, the 2's will cancel

import unittest
from pylab import rcParams
import matplotlib.animation as animation
from matplotlib import rc

# Uncomment if running on Jupyter
# %matplotlib inline

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def run_tests():
    unittest.main(argv=[''], verbosity=1, exit=False)

def loss(h, y):
    sq_error = (h - y) ** 3
    n = len(y)
    return 1.0 / (2 * n) * sq_error.sum()

class ManualLinearRegression:

    def predict(self, X):
        return np.dot(X, self._W)

    def _gradient_descent_step(self, X, targets, lr):

        predictions = self.predict(X)
        error = predictions - targets
        gradient = np.dot(X.T, error) / len(X)
        self._W -= lr * gradient

    def fit(self, X, y, n_iter=10000, lr=0.01):

        self._W = np.zeros(X.shape[1])

        # Used for showing line fitting over time
        self._cost_history = []
        self._w_history = [self._W]

        for i in range(n_iter):

            prediction = self.predict(X)
            cost = loss(prediction, y)
            self._cost_history.append(cost)
            self._gradient_descent_step(x, y, lr)
            self._w_history.append(self._W.copy())

        return self

class TestLoss(unittest.TestCase):

    def test_zero_h_zero_y(self):
        self.assertAlmostEqual(loss(h=np.array([0]), y=np.array([0])), 0)

    def test_one_h_zero_y(self):
        self.assertAlmostEqual(loss(h=np.array([1]), y=np.array([0])), 0.5)

    def test_two_h_zero_y(self):
        self.assertAlmostEqual(loss(h=np.array([2]), y=np.array([0])), 2)

    def test_zero_h_one_y(self):
        self.assertAlmostEqual(loss(h=np.array([0]), y=np.array([1])), 0.5)

    def test_zero_h_two_y(self):
        self.assertAlmostEqual(loss(h=np.array([0]), y=np.array([2])), 2)

    def test_find_coefficients(self):
        clf = ManualLinearRegression()
        clf.fit(x, y, n_iter=2000, lr=0.01)
        np.testing.assert_array_almost_equal(clf._W, np.array([180921.19555322,  56294.90199925]))

#####################################################################
# Data preprocessing
#
# Z-Score Formula
# x = (x - mu) / sigma
#
# Where:
# mu is population mean
# sigma is std deviation

x = training_data['GrLivArea'].values[:, np.newaxis]
y = training_data['SalePrice'].values

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x]

manual = ManualLinearRegression()
manual.fit(x, y, n_iter=2000, lr=0.01)

print(manual._W)

plt.title("Cost Function")
plt.xlabel("Num Iterations")
plt.ylabel("Cost")
plt.plot(manual._cost_history)
plt.show()

#####################################################################
# Line Fitting, Optimization, and Animation
# Plotting
from matplotlib import rc
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = plt.axes()
plt.title("Sale Price vs Living Area")
plt.xlabel("Living Area SqFt")
plt.ylabel("Sale Price")
plt.scatter(x[:, 1], y)
line, = ax.plot([], [], lw=2, color='pink')
annotation = ax.text(-1, 70000, '')
annotation.set_animated(True)

# Generate Animation
def init():
    line.set_data([], [])
    annotation.set_text('')
    return line, annotation

# Animation
def animate(i):
    x = np.linspace(-5, 20, 1000)
    y = manual._w_history[i][1]*x + manual._w_history[i][0]
    line.set_data(x, y)
    annotation.set_text("Cost = %.2f e10" % (manual._cost_history[i] / 10000000000))
    return line, annotation

anime = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=10, blit=True)

rc('animation', html='jshtml')
plt.draw()
plt.show()

#####################################################################
# Multivariable Regression
#
# We can use more complex formulas to allow our model to better fit
# more complex data
#
# Y(x1, x2, x3) = w1x1 + w2x2 + w3+x3 + w0

x = training_data[['GrLivArea', 'OverallQual', 'Fireplaces']]

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x]

manual = ManualLinearRegression()
manual.fit(x, y, n_iter=2000, lr=0.01)

print(manual._W)

plt.title("Cost Function")
plt.xlabel("Num Iterations")
plt.ylabel("Cost")
plt.plot(manual._cost_history)
plt.show()

fig = plt.figure()
ax = plt.axes()
plt.title("Living Area, Quality, Fireplaces")
plt.xlabel("Living Area SqFt")
plt.ylabel("Sale Price")
plt.scatter(x[:, 1], y)
line, = ax.plot([], [], lw=2, color='orange')
annotation = ax.text(-1, 70000, '')
annotation.set_animated(True)

anime = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=10, blit=True)

rc('animation', html='jshtml')
plt.draw()
plt.show()

