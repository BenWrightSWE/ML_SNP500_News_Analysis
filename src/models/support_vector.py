from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.usable_data import article_train, article_test, djia_train, djia_test
# just for my understanding, article is x and djia is y. different scalers for both

model = LinearSVC(C=0.3, loss="hinge")

model.fit(article_train, djia_train)

pred_train = model.predict(article_train)
pred_test = model.predict(article_test)

print("\nTraining Confusion Matrix & Classification Report:")
print(confusion_matrix(djia_train, pred_train))
print(classification_report(djia_train, pred_train))


print("\nTesting Classification Report:")
print(confusion_matrix(djia_test, pred_test))
print(classification_report(djia_test, pred_test))
