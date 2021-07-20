#####################################################################
# Statistics
#
# If you feel that you need an introduction to basic statistics,
# or if you just want to review what you've learned in the past.
#
# This guide attempts to show and explain starting with the very
# basic formulas and applications, and aims to give you greater
# insight when you return to, or move forward to, more advanced
# topics in Machine Learning or Data Mining
#
# We will provide examples on many of the formulas used in statistics
# how and where the formula may be applied,
# and then show how to perform these same functions using popular
# libraries, to save you time in the future
#
# But, understanding more about these basic operations should
# compliment your understanding of more compilcated topics
# we will encounter in other guides.
#
# Note:
# Many formulas are difficult to portray via text,
# I highly recommend you see them in a more traditional form
#
# A great example (that was heavily used in creating this guide)
# can be found here:
# cbmm.mit.edu/sites/default/files/documents/probability_handout.pdf
#
#####################################################################

#####################################################################
# The Basics
#
# For these examples:
# array: [1, 5, 5, 5, 20]
# n = length of array = 5
#
#####################################################################
# Sum: the total of all values in our data
# -> 1 + 5 + 5 + 5 + 20 = Sum = 36
#
#####################################################################
# Mean: (average) (mu)
# the Sum divided by the number of elements in our array
# -> 36 / 5 = Mean = 7.2
#
#####################################################################
# Median: (middle) (data is sorted)
#
# (Length of array is odd)
# (n + 1) / 2 = Median Index
#
# (Length of array is even)
# ( (n / 2) + ((n / 2) + 1) ) / 2
#
# -> Median  = (5 + 1) / 2 = 6 / 2 = 3
# the 3rd element in our array is 5 so our Median = 5
#
#####################################################################
# Mode: the value that occurs the most in the data
# in our example array, 5 occurs the most and is our Mode
#
#####################################################################
# Variance:
# provides an estimate of the average difference between each
# value and the mean
#
# (for population) (for each element (x) in the array)
# (sigma ^ 2) = (1 / n) * sum(x - mu)^2
#
# (for sample) (for each element (x) in the array)
# (s ^ 2) = (1 / (n - 1)) * sum(x - mean)^2
#
#####################################################################
# Standard Deviation:
# we will consider this to be the useable, or meaningful, form
# of Variance
#
# This measures the amount of Variance in the data
#
# Standard Deviation = the square root of Variance
#####################################################################
# What do they each tell us?
#
# Mean:
# is more useful when dealing with decently symmetrical data
# but can also indicate an influence of Outliers
#
# (An) Outlier:
# is data that is wildly different than the rest of the
# data, and can be (but is not always) the result of error
#
# For example, say we are looking at students grades and we have:
# 89, 72, 92, 94, 81, 12, 346, 95, 90, etc
#
# It should be clear in this situation that the number 346
# does not make sense and can have negative effects on our outcome
#
# Additionally, you may find situations where data that could be
# considered normal, such as the 12 above, is present
# But this may be the result of error, or could be meaningful
# to what you are looking for.
#
#
# Median:
# when the data is not symmetrical, the mean can be greatly
# influenced by outliers. In this situation the median can provide
# a better idea of the average or typical value
#
# However, in smaller datasets (around ~30 or less)
# the Sample Median tends to poorly estimate the Population Median
#
#
# Mode:
# is more useful when describing categorical or ordinal data
#
# Standard Deviation:
# a high deviation means each value is further from the mean
# a low deviation means each value is clustered closely to the mean
#
#####################################################################

#####################################################################
# Examples
# I am not using proper naming conventions here, so please do not
# emulate my poor behavior

array = [1, 5, 5, 5, 20]

def Sum(array):
    return sum(array)

def Mean(array):
    return (1 / len(array)) * sum(array)

# Mode is the value that occurs the most in the array
#
# Please be aware that this function does not account for arrays
# which may have multiple modes.
#
# *This will always return the mode with the highest value
def Mode(array):
    return max(array, key=array.count)

def Median(array):

    n = len(array)

    if n % 2 == 0:
        median_index = int(((n / 2) + ((n / 2) + 1)) / 2)
    else:
        median_index = int((n + 1) / 2)

    return array[median_index]

# Note: *we are using population variance
def Variance(array):
    return sum((val - Mean(array)) ** 2 for val in array) / len(array)

def StdDev(array):
    return (Variance(array)) ** (1 / 2)

# Print all information
def showBasics(array):

    print("""
Statistics:
    Sum: {0}
    Mean: {1}
    Mode: {2}
    Median: {3}
    Variance: {4}
    Standard Deviation: {5}\n""".format(Sum(array),
                                            Mean(array),
                                            Mode(array),
                                            Median(array),
                                            Variance(array),
                                            StdDev(array)))

# Show results
showBasics(array)

#####################################################################
# Moving Forward
#
# When we begin working with real data, undoubtedly it will be
# more sensible to use existing libraries
#
# Our goal up to this point was just to become familiar with a small
# number of simple statistical functions
#####################################################################
