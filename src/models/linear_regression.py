from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from src.usable_data import article_train, article_test, djia_train, djia_test
import numpy as np
import matplotlib.pyplot as plt

# scale tfidf vectors
scale = StandardScaler(with_mean=False)
trainInputScaled = scale.fit_transform(article_train)
testInputScaled = scale.transform(article_test)

# train  linear regression model
linearModel = LinearRegression()
linearModel.fit(trainInputScaled, djia_train)

# predictions
trainPred = linearModel.predict(trainInputScaled)
testPred = linearModel.predict(testInputScaled)

# evaluate model
print("linear regression results")
print("training r^2:", r2_score(djia_train, trainPred))
print("testing r^2:", r2_score(djia_test, testPred))

# classification labels 
threshold = np.mean(djia_train)

trainBin = (trainPred > threshold).astype(int)
testBin = (testPred > threshold).astype(int)

trainTrueBin = (djia_train > threshold).astype(int)
testTrueBin = (djia_test > threshold).astype(int)

# classification reports
print("\ntraining classification report:")
print(classification_report(trainTrueBin, trainBin))
print("training confusion matrix:")
print(confusion_matrix(trainTrueBin, trainBin))

print("\ntesting classification report:")
print(classification_report(testTrueBin, testBin))
print("testing confusion matrix:")
print(confusion_matrix(testTrueBin, testBin))

#  plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(djia_test, testPred, alpha=0.4)
plt.xlabel("actual djia change")
plt.ylabel("predicted djia change")
plt.title("linear regression: actual vs predicted")
plt.grid(True)
plt.show()

if __name__ == "__main__":
    linearRegModel("data/processed/snp_500.csv")
