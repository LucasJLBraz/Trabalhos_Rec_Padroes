import numpy as np


class KNN:
    """
    KNN (K-Nearest Neighbors) classifier.

    Parameters
    ----------
    X : np.ndarray
        Training data features of shape (n_samples, n_features).
    Y : np.ndarray
        Training data labels of shape (n_samples,).
    k : int (default=2)
        Number of nearest neighbors to use for classification.
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
    X : np.ndarray
        Training data features.
    Y : np.ndarray
        Training data labels.
    k : int
        Number of neighbors.
    minkowskiOrder : float
        Order of the Minkowski distance.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, k: int = 1, minkowskiOrder: float = 2) -> None:
        self.X = X
        self.Y = Y
        self.k = k
        self.minkowskiOrder = minkowskiOrder

    def _distMinkowski(self, X_i) -> np.ndarray:
        distances = np.sum(np.abs(self.X - X_i) ** self.minkowskiOrder, axis=1) ** (1.0 / self.minkowskiOrder)
        return distances

    def predict(self, X_new) -> np.ndarray:
        """
        Predicts the class labels for the given input samples using the k-nearest neighbors algorithm.

        Parameters
        ----------
        X_new : np.ndarray
            Input data to classify. Can be a 1D array representing a single sample or a 2D array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of predicted class labels for each input sample.
        """

        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        predictions = []
        for X_new_i in X_new:
            distances = self._distMinkowski(X_new_i)
            near_indices = np.argsort(distances)[:self.k]
            near_labels = self.Y[near_indices]

            values, counts = np.unique(near_labels, return_counts=True)
            classes_maj = values[np.argmax(counts)]
            predictions.append(classes_maj)
        return np.array(predictions)


class MDC:
    """
    MDC (Minimum Distance to Centroid) implements a simple centroid-based classification algorithm.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    Y : np.ndarray
        Array of class labels of shape (n_samples,).
    robustVersion : bool, optional
        If True, use the median instead of the mean to compute centroids (default is False).

    Attributes
    ----------
    X : np.ndarray
        Training feature matrix.
    Y : np.ndarray
        Training labels.
    robustVersion : bool
        Indicates whether to use the robust (median) version.
    num_centroids : int
        Number of unique classes (and centroids).
    centroids : np.ndarray or None
        Array of class centroids, computed after calling train().
    classes : np.ndarray
        Unique class labels.

    Methods
    -------
    train()
        Computes the centroids for each class using mean or median, depending on robustVersion.
    distEuclidiana(X_i)
        Computes the Euclidean distance from a sample X_i to all centroids.
    predict(X_new)
        Predicts the class labels for new samples X_new based on nearest centroid.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, robustVersion: bool = False) -> None:
        self.X = X
        self.Y = Y
        self.robustVersion = robustVersion
        self.num_centroids = len(np.unique(Y))
        self.centroids = None
        self.classes = np.unique(self.Y)

    def train(self) -> None:
        """
        Trains the classifier by computing the centroids for each class.

        For each unique class label in `self.classes`, this method calculates the centroid of the corresponding samples in `self.X`:
        - If `self.robustVersion` is False, the centroid is the mean of the samples.
        - If `self.robustVersion` is True, the centroid is the median of the samples.

        The resulting centroids are stored in `self.centroids` as a stacked NumPy array, where each row corresponds to a class centroid.
        """

        centroids = []
        for c in self.classes:
            class_indices = np.where(self.Y == c)[0]
            if not self.robustVersion:
                centroid = np.mean(self.X[class_indices, :], axis=0)
            else:
                centroid = np.median(self.X[class_indices, :], axis=0)
            centroids.append(centroid)
        self.centroids = np.vstack(centroids)

    def _distEuclidiana(self, X_i) -> np.ndarray:
        distances = np.sum((self.centroids - X_i) ** 2, axis=1) ** 0.5
        return distances

    def predict(self, X_new) -> np.ndarray:
        """
        Predicts the class labels for the given input samples.

        Parameters
        ----------
        X_new : np.ndarray
            Input data to classify. Can be a 1D array representing a single sample or a 2D array where each row is a sample.

        Returns
        -------
        np.ndarray
            Array of predicted class labels for each input sample.

        Notes
        -----
        This method computes the Euclidean distance between each input sample and the training data,
        assigns the label of the nearest neighbor to each sample, and returns the predicted labels.
        """

        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        predictions = []
        for X_new_i in X_new:
            distances = self._distEuclidiana(X_new_i)
            near_indice = np.argmin(distances)
            near_label = self.classes[near_indice]
            predictions.append(near_label)
        return np.array(predictions)


class MAXCO:
    """
    MAXCO (Maximum Cosine Correlation Classifier)

    A classifier that assigns samples to the class whose centroid (mean vector) has the highest cosine correlation
    with the sample, after mean-centering both vectors.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    Y : np.ndarray
        Target labels of shape (n_samples,).

    Attributes
    ----------
    X : np.ndarray
        Training feature matrix.
    Y : np.ndarray
        Training labels (flattened).
    classes : np.ndarray
        Unique class labels.
    centroids : np.ndarray or None
        Centroids (mean vectors) for each class, computed during training.

    Methods
    -------
    train()
        Computes and stores the centroids for each class based on the training data.
    predict(X_new)
        Predicts the class labels for new samples using maximum cosine correlation with class centroids.

    Private Methods
    ---------------
    _center(v)
        Returns the mean-centered version of vector v.
    _cos_corr(x, c, eps=1e-12)
        Computes the cosine correlation between mean-centered vectors x and c, with numerical stability.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = np.asarray(X)
        self.Y = np.asarray(Y).ravel()
        self.classes = np.unique(self.Y)
        self.centroids = None

    def train(self) -> None:
        """
        Trains the model by computing the centroid (mean feature vector) for each class.

        This method calculates the mean of the feature vectors (`self.X`) for each unique class label in `self.Y`.
        The resulting centroids are stored in `self.centroids` as a 2D NumPy array, where each row corresponds to a class centroid.

        Returns:
            None
        """
        centroids = []
        for c in self.classes:
            idx = self.Y == c
            centroids.append(self.X[idx].mean(axis=0))
        self.centroids = np.vstack(centroids)

    def _center(self, v: np.ndarray) -> np.ndarray:
        return v - np.mean(v)

    def _cos_corr(self, x: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> float:
        x0 = x - x.mean()
        c0 = c - c.mean()
        nx = np.linalg.norm(x0)
        nc = np.linalg.norm(c0)
        if nx < eps or nc < eps:
            return float("-inf")
        # correlação cos: produto interno dos vetores centralizados / (normas)
        return float(np.dot(x0, c0) / (nx * nc))

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for the provided input samples using the trained centroids.

        Parameters:
            X_new (np.ndarray): Input data to classify. Should be of shape (n_samples, n_features) or (n_features,).

        Returns:
            np.ndarray: Predicted class labels for each input sample.

        Raises:
            ValueError: If centroids have not been initialized (i.e., the model has not been trained).
        """
        # if self.centroids is None:
        #     self.train()
        if self.centroids is None:
            raise ValueError("Centroids não foram inicializados. Treine o Modelo.")
        Xn = np.atleast_2d(X_new)
        preds = []
        for x in Xn:
            rhos = [self._cos_corr(x, c) for c in self.centroids]
            k = int(np.argmax(rhos))
            preds.append(self.classes[k])
        return np.array(preds)
