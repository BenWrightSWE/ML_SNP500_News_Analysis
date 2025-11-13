from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score
from src.usable_data import article_train, article_test, djia_train, djia_test

# scale data for Ridge model

scale = StandardScaler(with_mean=False)
train_input_scaled = scale.fit_transform(article_train)
test_input_scaled = scale.transform(article_test)

# train and test Ridge model

for z in [0.1, 1, 10, 100, 1000, 100000, 100000000]:
    ridge = linear_model.Ridge(z)
    ridge.fit(train_input_scaled, djia_train)
    pred_train = ridge.predict(train_input_scaled)
    pred_test = ridge.predict(test_input_scaled)
    print(f"Hyperparameter Val: {z:.1f}")
    print("Coefficients:")
    print("Article_Vector")
    print(ridge.coef_)
    print()
    print("Train")
    print(f'R-Squared: {r2_score(djia_train, pred_train)}')
    print()
    print("Test")
    print(f'R-Squared: {r2_score(djia_test, pred_test)}')
    print("\n\n")