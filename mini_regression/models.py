import numpy as np


class BasicRegressionModel:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _initializing_weights(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def _predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def _update_weights(self, X, y, y_pred):
        n_samples = X.shape[0]
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        return dw, db


    def predict(self, X):
        return self._predict(X)


# 1. LR Model
class LinearRegressionModel(BasicRegressionModel):
    def train(self, X, y):
        n_samples, n_features = X.shape
        self._initializing_weights(n_features)

        # Applying gradient descent
        for _ in range(self.n_iterations):
            y_pred = self._predict(X)
            dw, db = self._update_weights(X, y, y_pred)

            # updating weighht and bias

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


class RidgeRegression(BasicRegressionModel):
    def __init__(self, alpha=0.1, learning_rate=0.01, n_iterations = 1000):
        super().__init__(learning_rate, n_iterations)
        self.alpha = alpha

    def train(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = self._linear_predict(X)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.alpha / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return self._linear_predict(X)

    def _linear_predict(self, X):
        return np.dot(X, self.weights) + self.bias



class LassoRegression(BasicRegressionModel):
    def __init__(self, alpha=0.01, learning_rate=0.001, n_iterations = 1000):
        super().__init__(learning_rate, n_iterations)
        self.alpha = alpha

    def train(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = self._linear_predict(X)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.alpha * np.sign(self.weights))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return self._linear_predict(X)

    def _linear_predict(self, X):
        return np.dot(X, self.weights) + self.bias


class SVR:
    def __init__(self, C = 1.0, epsilon = 0.1, learning_rate = 0.001, n_iterations = 1000):
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for ind, x_i in enumerate(X):
                cond = np.abs(np.dot(x_i, self.weights) + self.bias - y[ind]) >= self.epsilon
                if cond:
                    self.weights -= self.learning_rate * (2 * self.C * self.weights - np.dot(x_i, y[ind] - (np.dot(x_i, self.weights) + self.bias)))
                    self.bias -= self.learning_rate * (y[ind] - (np.dot(x_i, self.weights) + self.bias))

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class Node:
    def __init__(self, feature_index, threshold, left, right):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

class LeafNode:
    def __init__(self, value):
        self.value = value


class DecisionTreeReg:
    def __init__(self, max_depth = 10, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def train(self, X, y):
        self.tree = self._grow_tree(X,y)

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _grow_tree(self, X, y, depth = 0):
        n_samples, n_features = X.shape
        if n_samples >= self.min_samples_split and (self.max_depth is None or depth < self.max_depth):
            best_split = self._get_best_split(X, y, n_features)
            if best_split["var_reduction"] > 0:
                left_tree = self._grow_tree(best_split["X_left"], best_split["y_left"], depth + 1)
                right_tree = self._grow_tree(best_split["X_right"], best_split["y_right"], depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_tree, right_tree)

        return LeafNode(value=np.mean(y))

    def _get_best_split(self, X, y, n_features):
        best_split = {"var_reduction": -1}
        max_var_reduction = -float("inf")

        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                y_left, y_right = self._split(y, X_column, threshold)
                if len(y_left) > 0 and len(y_right) > 0:
                    curr_var_reduction = self._variance_reduction(y, y_left, y_right)
                    if curr_var_reduction > max_var_reduction:
                        max_var_reduction = curr_var_reduction
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "X_left": X[X_column <= threshold],
                            "y_left": y[X_column <= threshold],
                            "X_right": X[X_column > threshold],
                            "y_right": y[X_column > threshold],
                            "var_reduction": curr_var_reduction
                        }
        return best_split

    def _split(self, y, X_column, threshold):
        left_indices = np.where(X_column <= threshold)
        right_indices = np.where(X_column > threshold)
        return y[left_indices], y[right_indices]

    def _variance_reduction(self, y, y_left, y_right):
        var_total = np.var(y)
        var_left = np.var(y_left) * (len(y_left) / len(y))
        var_right = np.var(y_right) * (len(y_right) / len(y))
        return var_total - (var_left + var_right)

    def _predict(self, inputs, tree):
        if isinstance(tree, LeafNode):
            return tree.value
        if inputs[tree.feature_index] <= tree.threshold:
            return self._predict(inputs, tree.left)
        return self._predict(inputs, tree.right)


class RandomForestReg:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt", bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []

    def train(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.train(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        if self.bootstrap:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
        else:
            indices = np.arange(n_samples)
        max_features = self._get_max_features(X.shape[1])
        feature_indices = np.random.choice(X.shape[1], size=max_features, replace=False)
        return X[indices][:, feature_indices], y[indices]

    def _get_max_features(self, n_features):
        if isinstance(self.max_features, int):
            return min(n_features, self.max_features)
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        if self.max_features == "log2":
            return int(np.log2(n_features))
        return n_features


class GradientModel:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def initialize_params(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def gradientDescent(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_params(n_features)

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def batch_gradientDescent(self, X, y, batch_size):
        n_samples, n_features = X.shape
        self.initialize_params(n_features)

        for _ in range(self.n_iterations):
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                y_pred = np.dot(X_batch, self.weights) + self.bias
                dw = (1 / batch_size)* np.dot(X_batch.T, (y_pred - y_batch))
                db = (1 / batch_size)* np.sum(y_pred - y_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def stochastic_gradientDescent(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_params(n_features)

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                rand_idx = np.random.choice(n_samples)
                X_i = X[rand_idx, :].reshape(1, -1)
                y_i = y[rand_idx]
                y_pred = np.dot(X_i, self.weights) + self.bias
                dw = (1/ n_samples) * np.dot(X_i.T, (y_pred - y_i))
                db = (1 / n_samples) * np.sum(y_pred - y_i)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def mini_batch_gradientDescent(self, X, y, batch_size):
        n_samples, n_features = X.shape
        self.initialize_params(n_features)

        for _ in range(self.n_iterations):
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_pred = np.dot(X_batch, self.weights) + self.bias
                dw = (1 / batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1 / batch_size) * np.sum(y_pred - y_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []

    def train(self, X, y):
        residual = y

        for _ in range(self.n_estimators):
            tree = DecisionTreeReg(max_depth=self.max_depth)
            tree.train(X, residual)
            residual -= self.learning_rate * tree.predict(X)
            self.models.append(tree)

    def predict(self, X):
        predictions = np.sum([self.learning_rate * model.predict(X) for model in self.models], axis=0)
        return predictions
