# ============================================================
# SIMPLE LINEAR REGRESSION
# Dataset: Weight-Height
# ============================================================


# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 2. Load the dataset
# ------------------------------------------------------------

df = pd.read_csv("weight-height.csv")


# ------------------------------------------------------------
# 3. Display the dataset
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())


print("\nDataset shape:")
print(df.shape)


print("\nColumn names:")
print(df.columns)


print("\nDataset information:")
print(df.info())


print("\nStatistical summary:")
print(df.describe())


# ------------------------------------------------------------
# 4. Check missing values
# ------------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 5. Select Independent and Dependent variables
# ------------------------------------------------------------

# X = Independent variable
# Height is used to predict Weight

X = df[["Height"]]

# y = Dependent variable
y = df["Weight"]


print("\nIndependent Variable (X):")
print(X.head())


print("\nDependent Variable (y):")
print(y.head())


# ------------------------------------------------------------
# 6. Split data into training and testing data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining data size:")
print(X_train.shape)

print("\nTesting data size:")
print(X_test.shape)


# ------------------------------------------------------------
# 7. Create Linear Regression model
# ------------------------------------------------------------

model = LinearRegression()


# ------------------------------------------------------------
# 8. Train the model
# ------------------------------------------------------------

model.fit(X_train, y_train)


print("\nModel trained successfully!")


# ------------------------------------------------------------
# 9. Model parameters
# ------------------------------------------------------------

print("\nSlope (Coefficient):")
print(model.coef_[0])

print("\nIntercept:")
print(model.intercept_)


# ------------------------------------------------------------
# 10. Linear Regression equation
# ------------------------------------------------------------

print("\nLinear Regression Equation:")

print(
    f"Weight = {model.coef_[0]:.4f} * Height + "
    f"{model.intercept_:.4f}"
)


# ------------------------------------------------------------
# 11. Make predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)


print("\nActual vs Predicted values:")

comparison = pd.DataFrame({
    "Actual Weight": y_test.values,
    "Predicted Weight": y_pred
})

print(comparison.head(10))


# ------------------------------------------------------------
# 12. Evaluate the model
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\nModel Evaluation:")
print("-------------------------")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ------------------------------------------------------------
# 13. Visualize the data
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    y,
    alpha=0.6,
    label="Actual Data"
)

plt.xlabel("Height")
plt.ylabel("Weight")

plt.title("Height vs Weight")

plt.legend()

plt.show()


# ------------------------------------------------------------
# 14. Plot regression line
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    alpha=0.6,
    label="Actual Data"
)

plt.plot(
    X_test,
    y_pred,
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Height")
plt.ylabel("Weight")

plt.title("Simple Linear Regression")

plt.legend()

plt.show()


# ------------------------------------------------------------
# 15. Predict weight for a new height
# ------------------------------------------------------------

new_height = [[70]]

predicted_weight = model.predict(new_height)


print("\nNew Prediction:")
print("-------------------------")

print(
    f"For height = {new_height[0][0]}, "
    f"predicted weight = {predicted_weight[0]:.2f}"
)


# ------------------------------------------------------------
# END
# ------------------------------------------------------------

print("\nSimple Linear Regression completed successfully!")