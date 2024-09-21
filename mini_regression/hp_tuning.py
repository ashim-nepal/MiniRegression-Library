from .metrics import get_mse


def grid_search(_model, param_grid, X_train, y_train):

    best_params = {}
    best_score = float("inf")

    for params in param_grid:
        model = _model(**params)
        model.train(X_train, y_train)
        y_pred = model.predict(X_train)
        score = get_mse(y_train, y_pred)

        if score < best_score:
            best_score = score
            best_params = params

        return best_params, best_score
