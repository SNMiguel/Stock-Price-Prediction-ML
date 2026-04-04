"""
Ensemble model — stacks base model predictions using a Ridge meta-learner.
Trained via out-of-fold (OOF) predictions to avoid leakage into the meta-layer.
Also provides a confidence score based on base model agreement.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class EnsembleModel:
    """
    Stacking ensemble over the four base models.

    Base models are passed in already trained. The ensemble fits a
    Ridge meta-learner on their out-of-fold predictions so it learns
    how to weight and correct each model's output.
    """

    def __init__(self, base_models: dict):
        """
        Args:
            base_models: Dict mapping model name → fitted model object.
                         Each model must implement .predict(X) -> ndarray.
                         Example:
                           {
                             'Linear Regression': lr_model,
                             'Random Forest':     rf_model,
                             'SVR':               svr_model,
                             'Neural Network':    nn_model,
                           }
        """
        self.base_models  = base_models
        self.meta_learner = Ridge(alpha=1.0)
        self.is_fitted    = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            n_splits: int = 5) -> None:
        """
        Generate out-of-fold predictions and fit the meta-learner.

        Uses time-series KFold (no shuffle) so no future data leaks
        into any fold's training set.

        Args:
            X_train:  Training feature matrix.
            y_train:  Training target array.
            n_splits: Number of CV folds (default 5).
        """
        n_models = len(self.base_models)
        oof      = np.zeros((len(X_train), n_models))

        kf = KFold(n_splits=n_splits, shuffle=False)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val   = X_train[val_idx]

            for col, (name, model) in enumerate(self.base_models.items()):
                # Refit base model on fold subset
                model.fit(X_fold_train, y_fold_train)
                oof[val_idx, col] = model.predict(X_fold_val)

        # Fit meta-learner on stacked OOF predictions
        self.meta_learner.fit(oof, y_train)
        self.is_fitted = True
        print(f"✓ Ensemble fitted  ({n_splits} folds, {n_models} base models)")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions.

        Gets predictions from all base models, stacks them, then passes
        through the meta-learner.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Predictions array, shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        stacked = self._stack_predictions(X)
        return self.meta_learner.predict(stacked)

    def get_confidence(self, X: np.ndarray) -> float:
        """
        Return a confidence score for predictions on X.

        Confidence = 1 - normalised std of base model predictions.
        Higher value means models agree more → higher confidence.
        Returns a scalar in [0, 1].

        Args:
            X: Feature matrix (typically a single row for live trading).
        """
        stacked = self._stack_predictions(X)          # (n_samples, n_models)
        mean    = np.abs(stacked.mean(axis=1))
        std     = stacked.std(axis=1)

        # Avoid divide-by-zero when mean is near 0
        normalised_std = np.where(mean > 1e-6, std / mean, 0.0)
        confidence     = float(np.clip(1 - normalised_std.mean(), 0.0, 1.0))
        return confidence

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stack_predictions(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_models) array of base model predictions."""
        preds = []
        for model in self.base_models.values():
            p = model.predict(X)
            preds.append(p.flatten())
        return np.column_stack(preds)


if __name__ == "__main__":
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    np.random.seed(42)
    n, f = 300, 18
    X = np.random.randn(n, f)
    y = X[:, 0] * 10 + X[:, 1] * 5 + np.random.randn(n) * 2

    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Pre-fit base models
    base = {
        'Linear Regression': LinearRegression().fit(X_train, y_train),
        'Random Forest':     RandomForestRegressor(n_estimators=50,
                                                   random_state=42).fit(X_train, y_train),
        'SVR':               SVR(kernel='rbf').fit(X_train, y_train),
    }

    ensemble = EnsembleModel(base)
    ensemble.fit(X_train, y_train, n_splits=3)

    preds      = ensemble.predict(X_test)
    confidence = ensemble.get_confidence(X_test[:1])

    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"Ensemble RMSE      : {rmse:.4f}")
    print(f"Ensemble R²        : {r2:.4f}")
    print(f"Confidence (1 row) : {confidence:.4f}")
    print("models/ensemble.py: OK")
