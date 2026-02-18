import numpy as np
import matplotlib.pyplot as plt

# O objetivo do exemplo eh mostrar a propriedade de diagonalizacao
# da matriz de covariancia dos dados. Em outras palavras, verificar
# se a matriz de covariancia dos dados transformados Z eh diagonal
# e se as variancias sao iguais aos autovalores da matriz de covariancia
# Cx (dados originais).
#
# Autor: Guilherme Barreto
# Data: 17/05/2023

# Semente para reprodutibilidade (opcional)
# np.random.seed(0)

# Gera dados gaussianos com atributos nao-correlacionados
m1 = 5
sig1 = 1
m2 = -5
sig2 = sig1
m3 = 0
sig3 = sig1

N = 50000
X1 = np.random.normal(m1, sig1, N)
X2 = np.random.normal(m2, sig2, N)
X3 = np.random.normal(m3, sig3, N)

Xu = np.vstack([X1, X2, X3])

# Matriz desejada para os dados
Cd = np.array([[1, 1.8, -0.9], [1.8, 4, 0.6], [-0.9, 0.6, 9]])

# Decomposicao de Cholesky da matriz Cd
R = np.linalg.cholesky(Cd)

# Gera dados com atributos correlacionados com a matriz COV desejada
Xc = R.T @ Xu

# Aplicacao de PCA aos dados correlacionados
Cx = np.cov(Xc, bias=False)  # cada linha eh um atributo

# Autovalores e autovetores
L, V = np.linalg.eig(Cx)
idx = np.argsort(L)[::-1]
L = L[idx]
V = V[:, idx]

VEi = 100 * L / np.sum(L)
VEq = 100 * np.cumsum(L) / np.sum(L)

plt.figure()
plt.plot(VEq, "r-", linewidth=2)
plt.xlabel("Numero de autovalores principais (q)")
plt.ylabel("Variancia Explicada")
plt.gca().tick_params(labelsize=14)
plt.savefig("screeplot.eps", format="eps")

# PCA a partir da SVD
U3, S3, V3t = np.linalg.svd(Cx)

Q = V.T  # Matriz de transformacao (sem reducao de dimensionalidade)

Z = Q @ Xc

Cz = np.cov(Z, bias=False)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(Xc[0, :5000], Xc[1, :5000], "ro", linewidth=2)
plt.xlabel("Atributo X1")
plt.ylabel("Atributo X2")
plt.title("Dados Originais")
plt.gca().tick_params(labelsize=14)

plt.subplot(1, 2, 2)
plt.plot(Z[0, :5000], Z[1, :5000], "bo", linewidth=2)
plt.xlabel("Atributo Z1")
plt.ylabel("Atributo Z2")
plt.title("Dados Transformados")
plt.gca().tick_params(labelsize=14)

plt.savefig("pcademo.eps", format="eps")

# Reconstrucao dos dados originais
Xr = Q.T @ Z

E = Xc - Xr
NormaE2 = np.linalg.norm(E, "fro") ** 2
print("NormaE2 =", NormaE2)

E = E.reshape(-1, 1)
SSE = np.sum(E ** 2)
print("SSE =", SSE)
