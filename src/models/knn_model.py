from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from src.usable_data import article_train, article_test, djia_train, djia_test

#Scale data for KNN model
scale = StandardScaler(with_mean=False)
train_input_scaled = scale.fit_transform(article_train)
test_input_scaled = scale.transform(article_test)

#Train and evaluate KNN for multiple k values
best_k = None
best_acc = 0
accuracies = []
k_values = [1, 3, 5, 7, 9, 17, 23, 31]

for k in k_values:
    print(f"\nK Value: {k}")
    print("-" * 60)

    #Initialize and fit 
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_input_scaled, djia_train)

    #Predict
    pred_train = model.predict(train_input_scaled)
    pred_test = model.predict(test_input_scaled)

    train_acc = accuracy_score(djia_train, pred_train)
    test_acc = accuracy_score(djia_test, pred_test)
    accuracies.append(test_acc)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}\n")

    print("Train Classification Report:")
    print(classification_report(djia_train, pred_train, digits=4))

    print("Test Classification Report:")
    print(classification_report(djia_test, pred_test, digits=4))

    print("Confusion Matrix (Test):")
    print(confusion_matrix(djia_test, pred_test))

    print("=" * 60)

    #Best-performing k
    if test_acc > best_acc:
        best_acc = test_acc
        best_k = k

print(f"\nBest k = {best_k} with Test Accuracy = {best_acc:.4f}")

# Plot (accuracy vs. K)
plt.figure(figsize=(6, 4))
plt.plot(k_values, accuracies, marker='o')
plt.title("KNN Test Accuracy vs. K")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.show()
plt.close()