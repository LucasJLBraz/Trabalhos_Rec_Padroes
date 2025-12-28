import numpy as np


def mcovar1(X: np.ndarray) -> np.ndarray:
    """
    Calcula, com o metodo 1, a matriz de covariância de um conjunto de dados.

    Args:
        X (np.ndarray): Matriz de dados de entrada com dimensões (p, N), 
                        onde p é o número de características e N é o número de amostras.

    Returns:
        np.ndarry: Matriz de covariância de dimensões (p, p).
    """
    p, N = np.shape(X) # atributos x amostras
    soma = np.zeros((p, p)) # p x p 
    m = np.mean(X, 1) # media de cada atributo

    # Para cada amostra:
    for j in range(0, N):
        aux = X[:, j] - m # subtrai a media de cada atributo
        soma += np.outer(aux, aux) # Adiciona os produto do desvio
    return soma / N # Retorna a media


def mcovar2(X: np.ndarray) -> np.ndarray:
    """ 
    Calcula, com o metodo 2, a matriz de covariância de um conjunto de dados.

    Args:
        X (np.ndarray): Matriz de dados de entrada com dimensões (p, N), 
                        onde p é o número de características e N é o número de amostras.

    Returns:
        np.ndarry: Matriz de covariância de dimensões (p, p).
    """
    p, N = np.shape(X)
    m = np.mean(X, 1)
    aux = X - m[:, np.newaxis]

    C = (aux @ aux.T) / N
    return C


def mcovar3(X: np.ndarray) -> np.ndarray:
    """
    Calcula, com o metodo 3, a matriz de covariância de um conjunto de dados.

    Args:
        X (np.ndarray): Matriz de dados de entrada com dimensões (p, N), 
                        onde p é o número de características e N é o número de amostras.

    Returns:
        np.ndarry: Matriz de covariância de dimensões (p, p).
    """
    p, N = np.shape(X)
    R = np.zeros((p, p))
    m = np.mean(X, 1)
    for j in range(0, N):
        R = R + np.outer(X[:, j], X[:, j])
    C = R / N - np.outer(m, m)
    return C


def mcovar4(X: np.ndarray) -> np.ndarray:
    """
    Calcula, com o metodo 4, a matriz de covariância de um conjunto de dados.

    Args:
        X (np.ndarray): Matriz de dados de entrada com dimensões (p, N), 
                        onde p é o número de características e N é o número de amostras.

    Returns:
        np.ndarry: Matriz de covariância de dimensões (p, p).
    """
    p, N = np.shape(X)
    m = np.mean(X, 1)
    R = (X @ X.T) / N
    C = R - np.outer(m, m)
    return C