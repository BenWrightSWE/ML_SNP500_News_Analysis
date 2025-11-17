from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from scipy.optimize import minimize
import numpy as np

from src.usable_data import article_train, article_test, djia_train, djia_test

# scale tfidf inputs
scaler = StandardScaler(with_mean=False)
trainScale = scaler.fit_transform(article_train)
testScale = scaler.transform(article_test)

# convert sparse matrices to dense arrays
trainScale = trainScale.toarray()
testScale = testScale.toarray()

# prediction function for one row
def predictRow(xRow, weight, bias):
    return np.dot(weight, xRow) + bias

# ordinary least squares loss
def losses(params):
    weight = params[:-1]
    bias = params[-1]
    preds = trainScale @ weight + bias
    return np.sum((djia_train - preds) ** 2)

# initialize parameters for optimization
initParams = np.zeros(trainScale.shape[1] + 1)

# optimize parameters using bfgs
result = minimize(losses, initParams, method="BFGS")
params = result.x

weight = params[:-1]
bias = params[-1]

# make predictions
trainPred = trainScale @ weight + bias
testPred = testScale @ weight + bias

# show regression performance
print("\nmanual linear regression on vectorized news data (ols)")
print("training r^2:", r2_score(djia_train, trainPred))
print("testing r^2:", r2_score(djia_test, testPred))

# compute threshold using mean
threshold = np.mean(djia_train)

trainBin = (trainPred > threshold).astype(int)
testBin = (testPred > threshold).astype(int)

trainTrueBin = (djia_train > threshold).astype(int)
testTrueBin = (djia_test > threshold).astype(int)

# show classification results
print("\ntraining classification report:")
print(classification_report(trainTrueBin, trainBin))
print("training confusion matrix:")
print(confusion_matrix(trainTrueBin, trainBin))

print("\ntesting classification report:")
print(classification_report(testTrueBin, testBin))
print("testing confusion matrix:")
print(confusion_matrix(testTrueBin, testBin))
