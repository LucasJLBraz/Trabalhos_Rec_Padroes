import numpy as np


class KNN:
    """
    KNN (K-Nearest Neighbors) classificador.

    Parâmetros
    ----------
    X : np.ndarray
        Características dos dados de treinamento de forma (n_samples, n_features).
    Y : np.ndarray
        Rótulos dos dados de treinamento de forma (n_samples,).
    k : int (default=2)
        Número de vizinhos mais próximos a usar para classificação.
    minkowskiOrder : float, opcional (default=2)
        Ordem da distância de Minkowski (2 corresponde à distância Euclidiana).

    Métodos
    -------
    _predict(X_i)
        Computa a distância de Minkowski entre uma amostra X_i e todas as amostras de treinamento.
    predict(X_new)
        Prediz os rótulos de classe para os dados fornecidos.

    Atributos
    ----------
    X : np.ndarray
        Características dos dados de treinamento.
    Y : np.ndarray
        Rótulos dos dados de treinamento.
    k : int
        Número de vizinhos.
    minkowskiOrder : float
        Ordem da distância de Minkowski.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, k: int = 1, minkowskiOrder: float = 2) -> None:
        self.X = X
        self.Y = Y
        self.k = k
        self.minkowskiOrder = minkowskiOrder

    def _distMinkowski(self, X_i) -> np.ndarray:
        distancias = np.sum(np.abs(self.X - X_i) ** self.minkowskiOrder, axis=1) ** (1.0 / self.minkowskiOrder)
        return distancias

    def predict(self, X_new) -> np.ndarray:
        """
        Prediz os rótulos de classe para as amostras de entrada fornecidas usando o algoritmo dos k-vizinhos mais próximos.

        Parâmetros
        ----------
        X_new : np.ndarray
            Dados de entrada para classificar. Pode ser um array 1D representando uma única amostra ou um array 2D de forma (n_samples, n_features).

        Retorna
        -------
        np.ndarray
            Array dos rótulos de classe preditos para cada amostra de entrada.
        """

        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        predicoes = []
        for X_novo_i in X_new:
            distancias = self._distMinkowski(X_novo_i)
            indices_vizinhos = np.argsort(distancias)[:self.k]
            rotulos_vizinhos = self.Y[indices_vizinhos]

            valores, contagens = np.unique(rotulos_vizinhos, return_counts=True)
            classe_majoritaria = valores[np.argmax(contagens)]
            predicoes.append(classe_majoritaria)
        return np.array(predicoes)


class MDC:
    """
    MDC (Distância Mínima ao Centroide) implementa um algoritmo de classificação simples baseado em centroide.

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de características de forma (n_samples, n_features).
    Y : np.ndarray
        Array de rótulos de classe de forma (n_samples,).
    robustVersion : bool, opcional
        Se True, use a mediana em vez da média para calcular centroides (padrão é False).

    Atributos
    ----------
    X : np.ndarray
        Matriz de características de treinamento.
    Y : np.ndarray
        Rótulos de treinamento.
    robustVersion : bool
        Indica se deve usar a versão robusta (mediana).
    num_centroides : int
        Número de classes únicas (e centroides).
    centroides : np.ndarray ou None
        Array de centroides de classe, calculado após chamar train().
    classes : np.ndarray
        Rótulos de classe únicos.

    Métodos
    -------
    train()
        Calcula os centroides para cada classe usando média ou mediana, dependendo de robustVersion.
    _distEuclidiana(X_i)
        Calcula a distância Euclidiana de uma amostra X_i para todos os centroides.
    predict(X_new)
        Prediz os rótulos de classe para novas amostras X_new com base no centroide mais próximo.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, robustVersion: bool = False) -> None:
        self.X = X
        self.Y = Y
        self.robustVersion = robustVersion
        self.num_centroides = len(np.unique(Y))
        self.centroides = None
        self.classes = np.unique(self.Y)

    def train(self) -> None:
        """
        Treina o classificador calculando os centroides para cada classe.

        Para cada rótulo de classe único em `self.classes`, este método calcula o centroide das amostras correspondentes em `self.X`:
        - Se `self.robustVersion` for False, o centroide é a média das amostras.
        - Se `self.robustVersion` for True, o centroide é a mediana das amostras.

        Os centroides resultantes são armazenados em `self.centroides` como um array NumPy empilhado, onde cada linha corresponde a um centroide de classe.
        """

        centroides = []
        for c in self.classes:
            indices_classe = np.where(self.Y == c)[0]
            if not self.robustVersion:
                centroide = np.mean(self.X[indices_classe, :], axis=0)
            else:
                centroide = np.median(self.X[indices_classe, :], axis=0)
            centroides.append(centroide)
        self.centroides = np.vstack(centroides)

    def _distEuclidiana(self, X_i) -> np.ndarray:
        distancias = np.sum((self.centroides - X_i) ** 2, axis=1) ** 0.5
        return distancias

    def predict(self, X_new) -> np.ndarray:
        """
        Prediz os rótulos de classe para as amostras de entrada fornecidas.

        Parâmetros
        ----------
        X_new : np.ndarray
            Dados de entrada para classificar. Pode ser um array 1D representando uma única amostra ou um array 2D onde cada linha é uma amostra.

        Retorna
        -------
        np.ndarray
            Array dos rótulos de classe preditos para cada amostra de entrada.

        Notas
        -----
        Este método calcula a distância Euclidiana entre cada amostra de entrada e os dados de treinamento,
        atribui o rótulo do vizinho mais próximo a cada amostra e retorna os rótulos preditos.
        """

        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        predicoes = []
        for X_novo_i in X_new:
            distancias = self._distEuclidiana(X_novo_i)
            indice_vizinho = np.argmin(distancias)
            rotulo_vizinho = self.classes[indice_vizinho]
            predicoes.append(rotulo_vizinho)
        return np.array(predicoes)


class MAXCO:
    """
    MAXCO (Classificador de Máxima Correlação de Cosseno)

    Um classificador que atribui amostras à classe cujo centroide (vetor médio) tem a maior correlação de cosseno
    com a amostra, após centralizar a média de ambos os vetores.

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de características de forma (n_samples, n_features).
    Y : np.ndarray
        Rótulos alvo de forma (n_samples,).

    Atributos
    ----------
    X : np.ndarray
        Matriz de características de treinamento.
    Y : np.ndarray
        Rótulos de treinamento (achatados).
    classes : np.ndarray
        Rótulos de classe únicos.
    centroides : np.ndarray ou None
        Centroides (vetores médios) para cada classe, calculados durante o treinamento.

    Métodos
    -------
    train()
        Calcula e armazena os centroides para cada classe com base nos dados de treinamento.
    predict(X_new)
        Prediz os rótulos de classe para novas amostras usando máxima correlação de cosseno com centroides de classe.

    Métodos Privados
    ---------------
    _center(v)
        Retorna a versão centralizada em média do vetor v.
    _correlacao_cos(x, c, eps=1e-12)
        Calcula a correlação de cosseno entre vetores centralizados em média x e c, com estabilidade numérica.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = np.asarray(X)
        self.Y = np.asarray(Y).ravel()
        self.classes = np.unique(self.Y)
        self.centroids = None

    def train(self) -> None:
        """
        Treina o modelo calculando o centroide (vetor de característica média) para cada classe.

        Este método calcula a média dos vetores de características (`self.X`) para cada rótulo de classe único em `self.Y`.
        Os centroides resultantes são armazenados em `self.centroides` como um array NumPy 2D, onde cada linha corresponde a um centroide de classe.

        Retorna:
            None
        """
        centroides = []
        for c in self.classes:
            indice = self.Y == c
            centroides.append(self.X[indice].mean(axis=0))
        self.centroides = np.vstack(centroides)

    def _center(self, v: np.ndarray) -> np.ndarray:
        return v - np.mean(v)

    def _cos_corr(self, x: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> float:
        x0_centralizado = x - x.mean()
        c0_centralizado = c - c.mean()
        norma_x = np.linalg.norm(x0_centralizado)
        norma_c = np.linalg.norm(c0_centralizado)
        if norma_x < eps or norma_c < eps:
            return float("-inf")
        # correlação cosseno: produto interno dos vetores centralizados / (normas)
        return float(np.dot(x0_centralizado, c0_centralizado) / (norma_x * norma_c))

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Prediz os rótulos de classe para as amostras de entrada fornecidas usando os centroides treinados.

        Parâmetros:
            X_new (np.ndarray): Dados de entrada para classificar. Deve ser de forma (n_samples, n_features) ou (n_features,).

        Retorna:
            np.ndarray: Rótulos de classe preditos para cada amostra de entrada.

        Levanta:
            ValueError: Se os centroides não foram inicializados (ou seja, o modelo não foi treinado).
        """
        # if self.centroides is None:
        #     self.train()
        if self.centroides is None:
            raise ValueError("Centroides não foram inicializados. Treine o Modelo.")
        X_matriz = np.atleast_2d(X_new)
        predicoes = []
        for x in X_matriz:
            correlacoes = [self._cos_corr(x, c) for c in self.centroides]
            indice = int(np.argmax(correlacoes))
            predicoes.append(self.classes[indice])
        return np.array(predicoes)
