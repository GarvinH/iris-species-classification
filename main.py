import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#using iris.csv from kaggle: https://www.kaggle.com/uciml/iris

#load in data as string and shuffle data
data = np.genfromtxt("iris.csv", delimiter=",", dtype="str", skip_header=1)
x = data[:, 1:5].astype(float)
y = np.array(np.unique(data[:, 5], return_inverse=True)[1])

basic_model = LogisticRegression(solver="liblinear", random_state=0).fit(x,y)

print(basic_model.score(x,y))

x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.4)

x_cv, x_test, y_cv, y_test = train_test_split(x_cv, y_cv, test_size=0.2)

c = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
scores = np.zeros((len(c)))

for i in range(len(c)):
    model = LogisticRegression(solver="liblinear", random_state=0, C=c[i]).fit(x_train, y_train)
    scores[i] = model.score(x_cv, y_cv)

best_c = c[np.argmax(scores)]

test_model = LogisticRegression(solver="liblinear", random_state=0, C=best_c).fit(x_train, y_train)
# hypothesis = test_model.predict(x_test)

print(test_model.score(x_train, y_train))
print(test_model.score(x_cv, y_cv))
print(test_model.score(x_test, y_test))