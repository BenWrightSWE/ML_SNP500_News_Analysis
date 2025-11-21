from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score
from src.usable_data import article_train, article_test, djia_train, djia_test
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics

# scale data for Ridge model

scale = StandardScaler(with_mean=False)
train_input_scaled = scale.fit_transform(article_train)
test_input_scaled = scale.transform(article_test)

# train and test Ridge model

rtscores = []
rtscores_train = []
k_vals = [0.1, 1, 10, 100, 1000]

for k in k_vals:
    ridge = linear_model.Ridge(k)
    ridge.fit(train_input_scaled, djia_train)
    pred_train = ridge.predict(train_input_scaled)
    pred_test = ridge.predict(test_input_scaled)
    print(f"Hyperparameter Val: {k:.1f}")
    print("Coefficients:")
    print("Article_Vector")
    print(ridge.coef_)
    print()
    print("Train")
    print(f'R-Squared: {r2_score(djia_train, pred_train)}')
    rtscores_train.append(round(r2_score(djia_train, pred_train), 2))
    print()
    print("Test")
    print(f'R-Squared: {r2_score(djia_test, pred_test)}')
    rtscores.append(round(r2_score(djia_test, pred_test),2))
    print("\n\n")

    # allows the data to be used with the classification report
    pred_test_binary = (pred_test > 0).astype(int)
    djia_test_binary = (djia_test > 0).astype(int)
    print(f"Model with k = {k}\n")
    print("Classification Report \n")
    print(metrics.classification_report(djia_test_binary, pred_test_binary))
    print("Confusion Matrix Report \n")
    print(metrics.confusion_matrix(djia_test_binary, pred_test_binary))
    print("\n-------------------------------------------------------------------\n")

k_val_strs = ["0.1", "1", "10", "100", "1000"]
matplotlib.use("MacOSX") # allows the plot to show on MacOSX computers, comment out if not using/error occurs

plt.bar(k_val_strs, rtscores_train)
plt.title("R^2 for Ridge Model K Values Train")
plt.xlabel("k")
plt.ylabel("R_2")
plt.show()

plt.bar(k_val_strs, rtscores)
plt.title("R^2 for Ridge Model K Values Test")
plt.xlabel("k")
plt.ylabel("R_2")
plt.show()