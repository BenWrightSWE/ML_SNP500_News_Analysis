from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from scipy.optimize import minimize
import numpy as np

from src.usable_data import article_train, article_test, djia_train, djia_test

# scale the tfidf data 
scale = StandardScaler(with_mean=False)
train_scale = scale.fit_transform(article_train)
test_scale = scale.transform(article_test)

train_scale = train_scale.toarray()
test_scale = test_scale.toarray()

# prediction function (linear regression)
def pred_linear(x_row, weight, bias):
    return np.dot(weight, x_row) + bias

# ordinary least squares loss
def losses(parameters):
    weight = parameters[:-1]
    bias = parameters[-1]
    preds = np.dot(train_scale, weight) + bias
    return np.sum((djia_train - preds) ** 2)

# initialize parameters
init_val = np.zeros(train_scale.shape[1] + 1)

# optimize with BFGS (matches gradient-based notebooks)
optimize = minimize(losses, init_val, method="BFGS")
parameters = optimize.x

weight = parameters[:-1]
bias = parameters[-1]

# predictions
train_pred = np.dot(train_scale, weight) + bias
test_pred = np.dot(test_scale, weight) + bias

print("\nManual Linear Regression on Vectorized News Data (OLS)")
print("Training R^2:", r2_score(djia_train, train_pred))
print("Testing R^2:", r2_score(djia_test, test_pred))

# threshold using mean 
threshold = np.mean(djia_train)

train_bin = (train_pred > threshold).astype(int)
test_bin = (test_pred > threshold).astype(int)

train_true_bin = (djia_train > threshold).astype(int)
test_true_bin = (djia_test > threshold).astype(int)

print("\nTraining Classification Report:")
print(classification_report(train_true_bin, train_bin))
print("Training Confusion Matrix:")
print(confusion_matrix(train_true_bin, train_bin))

print("\nTesting Classification Report:")
print(classification_report(test_true_bin, test_bin))
print("Testing Confusion Matrix:")
print(confusion_matrix(test_true_bin, test_bin))
