import numpy as np
import os
from sklearn.linear_model import LogisticRegression

#using iris.csv from kaggle: https://www.kaggle.com/uciml/iris

#load in data as string and shuffle data
data = np.genfromtxt("iris.csv", delimiter=",", dtype="str", skip_header=1)
np.random.shuffle(data)
x = data[:, 1:5].astype(float)
y = np.array(np.unique(data[:, 5], return_inverse=True)[1])

model = LogisticRegression(solver="liblinear", random_state=0).fit(x,y)

print(model.score(x,y))