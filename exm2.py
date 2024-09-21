import numpy as np
import pandas as pd
from mini_regression.utils import train_test_split
from mini_regression.models import SVR, DecisionTreeReg, GradientBoostingRegressor
from mini_regression.metrics import get_mse, get_mae, get_rmse

df = pd.read_csv("admission.csv")
X = df.drop("Chance_of_Admit_", axis=1)
y = df["Chance_of_Admit_"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

svr_model = SVR()
decision_tree = DecisionTreeReg()
gradient_boosting = GradientBoostingRegressor()

svr_model.train(X_train, y_train)
decision_tree.train(X_train, y_train)
gradient_boosting.train(X_train, y_train)


y_pred_svr = svr_model.predict(X_test)
y_pred_tree = decision_tree.predict(X_test)
y_pred_gb = gradient_boosting.predict(X_test)

print("SVR MSE:", get_mse(y_test, y_pred_svr))
print("Decision Tree MSE:", get_mse(y_test, y_pred_tree))
print("Gradient Boosting MSE:", get_mse(y_test, y_pred_gb))


from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


sk_svr = SVR(C = 1.0, epsilon = 0.1)
sk_tree = DecisionTreeRegressor(max_depth = 10, min_samples_split = 2)
sk_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

sk_svr.fit(X_train, y_train)
sk_tree.fit(X_train, y_train)
sk_gb.fit(X_train, y_train)

y_pred_sk_svr = sk_svr.predict(X_test)
y_pred_sk_tree = sk_tree.predict(X_test)
y_pred_sk_gb = sk_gb.predict(X_test)

print("Sklearn SVR MSE:", mean_squared_error(y_test, y_pred_sk_svr))
print("Sklearn Decision Tree MSE:", mean_squared_error(y_test, y_pred_sk_tree))
print("Sklearn Gradient Boosting MSE:", mean_squared_error(y_test, y_pred_sk_gb))


