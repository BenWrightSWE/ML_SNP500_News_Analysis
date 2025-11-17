from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
import numpy as np

# importing processed data
from src.usable_data import article_train, article_test, djia_train, djia_test

# scale tfidf vectors
scaler = StandardScaler(with_mean=False)
trainScale = scaler.fit_transform(article_train)
testScale = scaler.transform(article_test)

# convert sparse matrices to dense numpy arrays
trainScale = trainScale.toarray()
testScale = testScale.toarray()

# train linear regression model using scikit (fast)
model = LinearRegression()
model.fit(trainScale, djia_train)

# make predictions
trainPred = model.predict(trainScale)
testPred = model.predict(testScale)

# extract learned weights and bias
weight = model.coef_
bias = model.intercept_

# print regression performance
print("\nmanual linear regression on vectorized news data (sklearn)")
print("training r^2:", r2_score(djia_train, trainPred))
print("testing r^2:", r2_score(djia_test, testPred))

# compute threshold using mean
threshold = np.mean(djia_train)

trainBin = (trainPred > threshold).astype(int)
testBin = (testPred > threshold).astype(int)

trainTrueBin = (djia_train > threshold).astype(int)
testTrueBin = (djia_test > threshold).astype(int)

# print classification results
print("\ntraining classification report:")
print(classification_report(trainTrueBin, trainBin))
print("training confusion matrix:")
print(confusion_matrix(trainTrueBin, trainBin))

print("\ntesting classification report:")
print(classification_report(testTrueBin, testBin))
print("testing confusion matrix:")
print(confusion_matrix(testTrueBin, testBin))
