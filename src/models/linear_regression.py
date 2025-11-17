import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def linearRegModel(csvPath):
    
    # load
    df = pd.read_csv(csvPath)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2013-05-01") & (df["date"] <= "2016-05-02")]

    # features
    df["priceChange"] = df["close"] - df["open"]
    df["priceRange"] = df["high"] - df["low"]
    df["percentChange"] = (df["close"] - df["open"]) / df["open"]
    df["upDown"] = (df["priceChange"] > 0).astype(int)

    #  prep x and y variables
    dropCols = ["date", "ticker", "close", "open", "high", "low", "upDown"]
    X = df.drop(columns=dropCols)
    y = df["priceChange"]

    #  test split
    testSizes = [0.35, 0.25, 0.45]

    for ts in testSizes:

        print(f"\nTrain/Test Split = {ts}\n")

        # split data
        XTrain, XTest, yTrain, yTest = train_test_split(
            X, y, test_size=ts, random_state=42
        )

        # scale
        scaler = StandardScaler()
        XTrainScaled = scaler.fit_transform(XTrain)
        XTestScaled = scaler.transform(XTest)

        # model
        model = LinearRegression()
        model.fit(XTrainScaled, yTrain)

        # predict
        yPred = model.predict(XTestScaled)

        # metrics
        r2 = r2_score(yTest, yPred)
        mse = mean_squared_error(yTest, yPred)

        print("Linear Regression Result")
        print("R² Score:", r2)
        print("Mean Squared Error:", mse)

        # plot for the actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(yTest, yPred, color="blue", alpha=0.5)
        plt.xlabel("Actual Price Change")
        plt.ylabel("Predicted Price Change")
        plt.title(f"Actual vs Predicted (Split = {ts})")
        plt.grid(True, linewidth=1.5)
        plt.show()

        # plot for the residuals
        residuals = yTest - yPred

        plt.figure(figsize=(8, 6))
        plt.scatter(yPred, residuals, color="blue", alpha=0.5)
        plt.axhline(0, color="red", linewidth=1.5)
        plt.xlabel("Predicted Price Change")
        plt.ylabel("Residuals")
        plt.title(f"Residual Plot (Split = {ts})")
        plt.grid(True, linewidth=1.5)
        plt.show()


if __name__ == "__main__":
    linearRegModel("data/processed/snp_500.csv")
