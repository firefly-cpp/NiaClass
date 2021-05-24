import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from niaclass import NiaClass

"""
This example presents how to use the NiaClass classifier instance.
The instantiated NiaClass will try to find the best set of classification rules for a dataset on the input.
"""

# read data from a randomly generated csv dataset without header row
# the last column in the dataset represents expected classification results
src = os.path.dirname(os.path.abspath(__file__)) + "/example_files/dataset.csv"
data = pd.read_csv(src, header=None)
y = data.pop(data.columns[len(data.columns) - 1])
x = data

# split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# instantiate NiaClass classifier with population size of 90 and 5000 evaluations
# it is going to use accuracy as fitness function and DifferentialEvolution as optimization algorithm
nc = NiaClass(90, 5000, "accuracy", "DifferentialEvolution")
# if you wish to set parameters of the optimization algorithm from the NiaPy framework, you can specify their values at the end of the constructor:
# NiaClass(90, 5000, 'accuracy', 'FireflyAlgorithm', alpha=0.5, betamin=0.2, gamma=1.0)

# fit classifier on training dataset
nc.fit(x_train, y_train)

# predict classes of individuals in the training set
y_predicted = nc.predict(x_test)

# print prediction accuracy to the standard output
print(accuracy_score(y_test, y_predicted))
