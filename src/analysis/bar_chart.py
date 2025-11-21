import matplotlib.pyplot as plt

testing_accuracies = [.50, .53, .53, .5296]
models = ['Linear Regression', 'Ridge Regression', 'Support Vector Classifier', 'K-Nearest Neighbors']

plt.figure(figsize=(10, 6))

plt.bar(models, testing_accuracies)
plt.ylim(0, 1)
plt.xlabel('Models')
plt.ylabel('Testing Split Accuracies')
plt.title('Testing Accuracies for each Model Used')

plt.show()