import numpy as np
import pandas as pd
from mini_regression.utils import train_test_split
from mini_regression.models import GradientModel, LinearRegressionModel
from mini_regression.metrics import get_mse, get_mae, get_rmse

df = pd.read_csv("admission.csv")
X = df.drop("Chance_of_Admit_", axis=1)
y = df["Chance_of_Admit_"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

model = GradientModel()

print("Testing Standard Gradient Descent...")
model.mini_batch_gradientDescent(X_train, y_train, 32)
y_pred_standard_gd = model.predict(X_test)
mse_standard_gd = get_mse(y_test, y_pred_standard_gd)
print("Standard GD MSE:", mse_standard_gd)