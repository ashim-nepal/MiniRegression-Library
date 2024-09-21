import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn as sk
from sklearn.metrics import mean_squared_error

from mini_regression.models import LinearRegressionModel, LassoRegression, RidgeRegression, DecisionTreeReg, RandomForestReg
import mini_regression as mr
from mini_regression.metrics import get_mse
from mini_regression.utils import train_test_split

# X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
# y = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
df= pd.read_csv("admission.csv")
X = df.drop("Chance_of_Admit_", axis=1)
y = df["Chance_of_Admit_"]


X_sk = sk.preprocessing.StandardScaler()
X_sk = X_sk.fit_transform(X)

X_mr = mr.preprocessing.StandardScaler()
X_mr = X_mr.fit_transform(X)

X_sk_train, X_sk_test, y_train, y_test = train_test_split(X_sk, y)
lr_sk = LinearRegression()
lr_sk.fit(X_sk_train, y_train)
prediction_lr_sk = lr_sk.predict(X_sk_test)
print(f"MSE(SKlearn) for SK LR model {mean_squared_error(y_test, prediction_lr_sk)}")
print(f"MSE(MiniReg) for SK LR model {get_mse(y_test, prediction_lr_sk)}")

X_mr_train, X_mr_test, y_train, y_test = train_test_split(X_mr, y)
lr_mr = LinearRegressionModel()
lr_mr.train(X_mr_train, y_train)
prediction_lr_mr = lr_mr.predict(X_mr_test)
print(f"MSE(Sklearn) for MR LR model {mean_squared_error(y_test, prediction_lr_mr)}")
print(f"MSE(MiniReg) for MR LR model {get_mse(y_test, prediction_lr_mr)}")


# Checking the lasso Regression of both the sklearn and mini_Revression


# X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
# y = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
X = df.drop("Chance_of_Admit_", axis=1)
y = df["Chance_of_Admit_"]


X_sk = sk.preprocessing.StandardScaler()
X_sk = X_sk.fit_transform(X)

X_mr = mr.preprocessing.StandardScaler()
X_mr = X_mr.fit_transform(X)

X_sk_train, X_sk_test, y_train, y_test = train_test_split(X_sk, y)
lasso_sk = Lasso(alpha=0.1)
lasso_sk.fit(X_sk_train, y_train)
prediction_sk_lasso = lasso_sk.predict(X_sk_test)
print(f"MSE(SKlearn) for SK Lasso model {mean_squared_error(y_test, prediction_sk_lasso)}")
print(f"MSE(MiniReg) for SK Lasso model {get_mse(y_test, prediction_sk_lasso)}")

X_mr_train, X_mr_test, y_train, y_test = train_test_split(X_mr, y)
lasso_mr = LassoRegression()
lasso_mr.train(X_mr_train, y_train)
prediction_mr_lasso = lasso_mr.predict(X_mr_test)
print(f"MSE(SKlearn) for MR Lasso model {mean_squared_error(y_test, prediction_mr_lasso)}")
print(f"MSE(MiniReg) for MR Lasso model {get_mse(y_test, prediction_mr_lasso)}")

# Ridge Regression

# X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
# y = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
X = df.drop("Chance_of_Admit_", axis=1)
y = df["Chance_of_Admit_"]

X_sk = sk.preprocessing.StandardScaler()
X_sk = X_sk.fit_transform(X)

X_mr = mr.preprocessing.StandardScaler()
X_mr = X_mr.fit_transform(X)

X_sk_train, X_sk_test, y_train, y_test = train_test_split(X_sk, y)
ridge_sk = Ridge(alpha=0.03)
ridge_sk.fit(X_sk_train, y_train)
prediction_sk_ridge = ridge_sk.predict(X_sk_test)
print(f"MSE(SKlearn) for SK Ridge model {mean_squared_error(y_test, prediction_sk_ridge)}")
print(f"MSE(MiniReg) for SK Ridge model {get_mse(y_test, prediction_sk_ridge)}")

X_mr_train, X_mr_test, y_train, y_test = train_test_split(X_mr, y)
ridge_mr = RidgeRegression(alpha=0.03)
ridge_mr.train(X_mr_train, y_train)
prediction_mr_ridge = ridge_mr.predict(X_mr_test)
print(f"MSE(SKlearn) for MR Ridge model {mean_squared_error(y_test, prediction_mr_ridge)}")
print(f"MSE(MiniReg) for MR Ridge model {get_mse(y_test, prediction_mr_ridge)}")

# visualize_prediction(y_test, prediction_mr_ridge)
print("\n\n\n\n")


X = df.drop("Chance_of_Admit_", axis=1)
y = df["Chance_of_Admit_"]

X_sk = sk.preprocessing.StandardScaler()
X_sk = X_sk.fit_transform(X)

X_mr = mr.preprocessing.StandardScaler()
X_mr = X_mr.fit_transform(X)

X_sk_train, X_sk_test, y_train, y_test = train_test_split(X_sk, y)
desTree_sk = DecisionTreeRegressor()
desTree_sk.fit(X_sk_train, y_train)
prediction_sk_desTree = desTree_sk.predict(X_sk_test)
print(f"MSE(SKlearn) for SK desTree model {mean_squared_error(y_test, prediction_sk_desTree)}")
print(f"MSE(MiniReg) for SK desTree model {get_mse(y_test, prediction_sk_desTree)}")

X_mr_train, X_mr_test, y_train, y_test = train_test_split(X_mr, y)
desTree_mr = DecisionTreeReg()
desTree_mr.train(X_mr_train, y_train)
prediction_mr_desTree = desTree_mr.predict(X_mr_test)
print(f"MSE(SKlearn) for MR desTree model {mean_squared_error(y_test, prediction_mr_desTree)}")
print(f"MSE(MiniReg) for MR desTree model {get_mse(y_test, prediction_mr_desTree)}")



X = df.drop("Chance_of_Admit_", axis=1)
y = df["Chance_of_Admit_"]

X_sk = sk.preprocessing.StandardScaler()
X_sk = X_sk.fit_transform(X)

X_mr = mr.preprocessing.StandardScaler()
X_mr = X_mr.fit_transform(X)

X_sk_train, X_sk_test, y_train, y_test = train_test_split(X_sk, y)
randForest_sk = RandomForestRegressor()
randForest_sk.fit(X_sk_train, y_train)
prediction_sk_randForest = randForest_sk.predict(X_sk_test)
print(f"MSE(SKlearn) for SK randForest model {mean_squared_error(y_test, prediction_sk_randForest)}")
print(f"MSE(MiniReg) for SK randForest model {get_mse(y_test, prediction_sk_randForest)}")

X_mr_train, X_mr_test, y_train, y_test = train_test_split(X_mr, y)
randForest_mr = RandomForestReg()
randForest_mr.train(X_mr_train, y_train)
prediction_mr_randForest = randForest_mr.predict(X_mr_test)
print(f"MSE(SKlearn) for MR randForest model {mean_squared_error(y_test, prediction_mr_randForest)}")
print(f"MSE(MiniReg) for MR randForest model {get_mse(y_test, prediction_mr_randForest)}")
