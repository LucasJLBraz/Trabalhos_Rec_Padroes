import numpy as np


class KNN:
    """
    KNN (K-Nearest Neighbors) classifier.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to use for classification.
    X : np.ndarray
        Training data features of shape (n_samples, n_features).
    Y : np.ndarray
        Training data labels of shape (n_samples,).
    minkowskiOrder : float, optional (default=2)
        Order of the Minkowski distance (2 corresponds to Euclidean distance).

    Methods
    -------
    _predict(X_i)
        Computes the Minkowski distance between a given sample X_i and all training samples.
    predict(X_new)
        Predicts the class labels for the provided data.

    Attributes
    ----------
    k : int
        Number of neighbors.
    X : np.ndarray
        Training data features.
    Y : np.ndarray
        Training data labels.
    minkowskiOrder : float
        Order of the Minkowski distance.

    Examples #TODO: Improve example usage.
    --------
    >>> X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
    >>> Y_train = np.array([0, 0, 1, 1])
    >>> knn = KNN(k=3, X=X_train, Y=Y_train)
    >>> X_test = np.array([[2, 2], [5, 5]])
    >>> knn.predict(X_test)
    array([0, 1])
    """
    def __init__(self, k: int, X: np.ndarray, Y: np.ndarray, minkowskiOrder: float = 2) -> None:
        self.k = k
        self.X = X
        self.Y = Y
        self.minkowskiOrder = minkowskiOrder

    def _distMinkowski(self, X_i) -> np.ndarray:
        distances = np.sum(np.abs(self.X - X_i) ** self.minkowskiOrder, axis=1) ** (1.0 / self.minkowskiOrder)
        return distances

    def predict(self, X_new) -> np.ndarray:
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        predictions = []
        for X_new_i in X_new:
            distances = self._distMinkowski(X_new_i)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.Y[nearest_indices]

            values, counts = np.unique(nearest_labels, return_counts=True)
            majority_label = values[np.argmax(counts)]
            predictions.append(majority_label)
        return np.array(predictions)


class DMC:
    def __init__(self) -> None:
        pass


class MAXCO:
    def __init__(self) -> None:
        pass