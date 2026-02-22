import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from collections import Counter
from classificao_padroes.models_trabalho2 import mcovar4


def analizar_invertibilidade(df, label):
    print(f"Análise de Invertibilidade: {label}")
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Matriz de Covariância Global
    cov_global = mcovar4(X.values.T)
    rank_g = np.linalg.matrix_rank(cov_global)
    cond_g = np.linalg.cond(cov_global)
    rcond_g = 1 / cond_g
    print(f"Global -> Posto: {rank_g}, Cond: {cond_g:.2e}, RCond: {rcond_g:.2e}")

    # Matrizes de Covariância por Classe
    for c in np.unique(y):
        X_c = X[y == c]
        cov_c = mcovar4(X_c.values.T)
        rank_c = np.linalg.matrix_rank(cov_c)
        cond_c = np.linalg.cond(cov_c)
        rcond_c = 1 / cond_c
        print(f"Classe {c} -> Posto: {rank_c}, Cond: {cond_c:.2e}, RCond: {rcond_c:.2e}")
    print("-" * 50)


class ClassificadorQuadraticoGaussiano:
    """
    Um classificador probabilístico que modela cada classe como uma distribuição gaussiana multivariada
    com matriz de covariância própria, permitindo fronteiras de decisão quadráticas entre classes.

    Attributes
    ----------
    classes_ : dict
        Array contendo os rótulos únicos das classes encontradas durante o treinamento.
    matrizes_cov : dict
        Dicionário armazenando a matriz de covariância para cada classe.
    medias_ : dict
        Dicionário armazenando o vetor de médias para cada classe.
    priors_ : dict
        Dicionário armazenando a probabilidade a priori (proporção) de cada classe.
    distancias : dict
        Dicionário para armazenar distâncias calculadas.

    Methods
    -------
    fit(X, y)
        Treina o classificador calculando as médias, matrizes de covariância e priors para cada classe.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dados de treinamento.
        y : array-like, shape (n_samples,)
            Rótulos das classes correspondentes aos dados de treinamento.

    predict(X)
        Classifica as amostras utilizando a distância de Mahalanobis e o critério de máxima verossimilhança.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dados a serem classificados.

        Returns
        -------
        predictions : array, shape (n_samples,)
            Rótulos das classes preditas para cada amostra.

        Notes
        -----
        Implementa tratamento de erro para matrizes singulares através de regularização (ridge).
        Utiliza o logaritmo do determinante e a distância de Mahalanobis para otimização numérica.
    """

    def __init__(self):
        self.classes_ = {}
        self.matrizes_cov = {}
        self.medias_ = {}
        self.priors_ = {}
        self.distancias = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = X_c.shape[0] / n_samples
            self.medias_[c] = np.mean(X_c, axis=0)
            self.matrizes_cov[c] = mcovar4(X_c.T)

    def predict(self, X):
        scores = []
        for c in self.classes_:
            diff = X - self.medias_[c]
            cov = self.matrizes_cov[c]

            # TRATAMENTO DE ERRO/REGULARIZAÇÃO
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # Fallback para pseudo-inversa ou adicionar regularização
                print(f"Aviso: Matriz singular na classe {c}, usando regularização.")
                reg = 1e-6 * np.eye(cov.shape[0])
                inv_cov = np.linalg.inv(cov + reg)

            # Distancia de Mahalanobis (Otimizada)
            # (N, p) @ (p, p) * (N, p) -> sum axis 1 -> (N,)
            mahalanobis = np.sum(diff @ inv_cov * diff, axis=1)

            # Log-determinante
            _, log_det = np.linalg.slogdet(cov)

            # Score para minimização
            score_c = mahalanobis + log_det - 2 * np.log(self.priors_[c])
            scores.append(score_c)

        scores = np.array(scores).T  # Transforma para (N_samples, N_classes)
        return self.classes_[np.argmin(scores, axis=1)]


def dunn_index(X, labels):
    """
    Calcula o Índice de Dunn: min(dist_inter_cluster) / max(diametro_intra_cluster).
    Quanto maior, melhor (clusters compactos e bem separados).
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0

    # Matriz de distâncias (O(N^2)) - Cuidado com datasets massivos
    distances = euclidean_distances(X)

    # 1. Calcular diâmetros (máxima distância intra-cluster)
    max_intra_dists = []
    for label in unique_labels:
        mask = (labels == label)
        if np.sum(mask) < 2:
            max_intra_dists.append(0.0)
            continue
        cluster_dists = distances[mask][:, mask]
        max_intra_dists.append(np.max(cluster_dists))

    max_intra = np.max(max_intra_dists) if max_intra_dists else 0.0

    if max_intra == 0:
        return 0.0  # Evita divisão por zero se todos clusters forem pontos únicos

    # 2. Calcular separação (mínima distância inter-cluster)
    min_inter_dists = []
    for i, label_i in enumerate(unique_labels):
        mask_i = (labels == label_i)
        for j, label_j in enumerate(unique_labels):
            if i >= j:
                continue  # Matriz simétrica, ignora diagonal e repetidos
            mask_j = (labels == label_j)
            inter_dists = distances[mask_i][:, mask_j]
            min_inter_dists.append(np.min(inter_dists))

    min_inter = np.min(min_inter_dists) if min_inter_dists else 0.0

    return min_inter / max_intra


def i_index(X, labels):
    """
    Calcula o Índice I (I-Index): (1/k) * (E_1/E_k) * D_k
    Onde:
        E_1 = distância total ao centroide global (k=1)
        E_k = soma das distâncias intra-cluster
        D_k = máxima distância inter-cluster (entre centroides)
    Quanto maior, melhor.
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2:
        return 0

    # Centroide global
    global_centroid = np.mean(X, axis=0)

    # E_1: soma das distâncias ao centroide global
    E_1 = np.sum(np.linalg.norm(X - global_centroid, axis=1))

    # E_k: soma das distâncias intra-cluster
    E_k = 0
    centroids = []
    for label in unique_labels:
        mask = (labels == label)
        cluster_points = X[mask]
        if len(cluster_points) == 0:
            continue
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        E_k += np.sum(np.linalg.norm(cluster_points - centroid, axis=1))

    centroids = np.array(centroids)

    if E_k == 0 or len(centroids) < 2:
        return 0

    # D_k: máxima distância entre centroides
    centroid_distances = euclidean_distances(centroids)
    D_k = np.max(centroid_distances)

    # Índice I
    I = (1.0 / k) * (E_1 / E_k) * D_k

    return I


def ball_hall_index(X, labels):
    """
    Calcula o Índice Ball-Hall: média da dispersão intra-cluster.
    Quanto menor, melhor (clusters compactos).
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2:
        return np.inf

    total_dispersion = 0
    for label in unique_labels:
        mask = (labels == label)
        cluster_points = X[mask]
        if len(cluster_points) == 0:
            continue
        centroid = np.mean(cluster_points, axis=0)
        dispersion = np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
        total_dispersion += dispersion

    return total_dispersion / k


class ClassificadorDMP:
    """
    Classificador de Distância Mínima ao Protótipo (DMP).
    """

    def __init__(self, k_min=2, k_max=10, n_runs=20):
        """
        Args:
            k_min (int): Mínimo de clusters a testar (>=2 para índices de validação).
            k_max (int): Máximo de clusters a testar.
            n_runs (int): Número de rodadas do K-Means (Passo 2/3).
        """
        self.k_min = max(2, k_min)  # Indices precisam de pelo menos 2 clusters
        self.k_max = k_max
        self.n_runs = n_runs
        self.prototipos_ = []
        self.labels_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.prototipos_ = []
        self.labels_ = []

        print(f"Treinando DMP Multi-Protótipo. Classes: {self.classes_}")

        # Passo 1: Separar os dados por classe
        for c in self.classes_:
            X_c = X[y == c]
            n_amostras = X_c.shape[0]

            # Proteção: Não podemos ter mais clusters que amostras
            # E precisamos de pelo menos k_min amostras
            limite_k = min(self.k_max, n_amostras - 1)

            if limite_k < self.k_min:
                print(f"  [Aviso] Classe {c}: Amostras insuficientes ({n_amostras}) para validação de clusters.")
                print("          Usando média simples (1 protótipo).")
                self.prototipos_.append(np.mean(X_c, axis=0))
                self.labels_.append(c)
                continue

            # Dicionário para "cachear" os modelos vencedores de cada K
            # Estrutura: {k: {'model': KMeansObject, 'db': float, 'ch': float, 'dn': float, ...}}
            candidatos = {}

            # Loop para variar K
            for k in range(self.k_min, limite_k + 1):
                best_inertia = np.inf
                melhorModeloK = None

                # Passo 2 e 3: Rodar K-Means Nr vezes e escolher o de menor inércia (SSD)
                # Isso garante que estamos avaliando o "melhor" K-Means possível para este k
                for _ in range(self.n_runs):
                    kmeans = KMeans(n_clusters=k, n_init=1, init='k-means++')
                    kmeans.fit(X_c)

                    if kmeans.inertia_ < best_inertia:
                        best_inertia = kmeans.inertia_
                        melhorModeloK = kmeans  # Guarda o objeto treinado

                # Passo 4: Calcular índices de validação para o modelo vencedor deste K
                labels = melhorModeloK.labels_  # type: ignore

                # Armazena tudo no dicionário de candidatos
                candidatos[k] = {
                    'model': melhorModeloK,
                    'db': davies_bouldin_score(X_c, labels),         # Menor é melhor
                    'ch': calinski_harabasz_score(X_c, labels),      # Maior é melhor
                    'dunn': dunn_index(X_c, labels),                 # Maior é melhor
                    'silhouette': silhouette_score(X_c, labels),     # Maior é melhor
                    'i_index': i_index(X_c, labels),                 # Maior é melhor
                    'ball_hall': ball_hall_index(X_c, labels)        # Menor é melhor
                }

            # Passo 5: Seleção do K ótimo baseado na MODA (k mais frequente)
            # Extrair listas para buscar min/max de cada índice
            ks_tested = list(candidatos.keys())

            # Para cada índice, determinar qual k é o melhor
            votos = []

            # Davies-Bouldin: menor é melhor
            dbs = [candidatos[k]['db'] for k in ks_tested]
            k_best_db = ks_tested[np.argmin(dbs)]
            votos.append(k_best_db)

            # Calinski-Harabasz: maior é melhor
            chs = [candidatos[k]['ch'] for k in ks_tested]
            k_best_ch = ks_tested[np.argmax(chs)]
            votos.append(k_best_ch)

            # Dunn: maior é melhor
            dns = [candidatos[k]['dunn'] for k in ks_tested]
            k_best_dn = ks_tested[np.argmax(dns)]
            votos.append(k_best_dn)

            # Silhouette: maior é melhor
            silhouettes = [candidatos[k]['silhouette'] for k in ks_tested]
            k_best_sil = ks_tested[np.argmax(silhouettes)]
            votos.append(k_best_sil)

            # I-Index: maior é melhor
            i_indices = [candidatos[k]['i_index'] for k in ks_tested]
            k_best_i = ks_tested[np.argmax(i_indices)]
            votos.append(k_best_i)

            # Ball-Hall: menor é melhor
            ball_halls = [candidatos[k]['ball_hall'] for k in ks_tested]
            k_best_bh = ks_tested[np.argmin(ball_halls)]
            votos.append(k_best_bh)

            # Escolher o k mais frequente (MODA)
            qtd_votos = Counter(votos)
            k_opt = qtd_votos.most_common(1)[0][0]  # k com maior frequência

            # Informação detalhada para debug
            votos_str = f"DB={k_best_db}, CH={k_best_ch}, Dunn={k_best_dn}, Sil={k_best_sil}, I={k_best_i}, BH={k_best_bh}"
            freq_str = ", ".join([f"k={k}({count}x)" for k, count in qtd_votos.most_common()])
            print(f"  Classe {c}: Votos [{votos_str}] -> Frequências: [{freq_str}] -> K ótimo: {k_opt}")

            # Passo 6: Recuperar os protótipos do modelo JÁ TREINADO
            modelo_vencedor = candidatos[k_opt]['model']

            for centroide in modelo_vencedor.cluster_centers_:
                self.prototipos_.append(centroide)
                self.labels_.append(c)

        # Converter para numpy array para eficiência no predict
        self.prototipos_ = np.array(self.prototipos_)
        self.labels_ = np.array(self.labels_)

    def predict(self, X):
        """
        Classifica cada amostra em X com base no protótipo mais próximo (Euclidiana).
        """
        if len(self.prototipos_) == 0:
            raise ValueError("O modelo não foi treinado ou não gerou protótipos.")

        # Calcula distância de todas as amostras para todos os protótipos (Matriz N x N_prototypes)
        dists = cdist(X, self.prototipos_, metric='euclidean')

        # Índice do protótipo com menor distância
        idx_proto_proximo = np.argmin(dists, axis=1)

        # Recupera o rótulo da classe associado àquele protótipo
        return self.labels_[idx_proto_proximo]


class PCA:
    """
    Principal Component Analysis (PCA) via SVD da Matriz de Covariância.
    """

    def __init__(self, n_componentes=None):
        """
        n_components:
            - Se float (0 < n < 1): Seleciona componentes para explicar 'n' % da variância.
            - Se int (n >= 1): Seleciona exatamente 'n' componentes.
            - Se None: Mantém todos os componentes.
        """
        self.n_componentes = n_componentes
        self.componentes_ = None      # Matriz de projeção (W)
        self.media_ = None            # Média para centralização
        self.variancia_explicada_ = None       # Autovalores (lambda)
        self.razao_variancia_explicada_ = None  # Variância explicada (%)

    def fit(self, X):
        # Passo 1: Calcular a média de cada atributo
        self.media_ = np.mean(X, axis=0)

        # Passo 2: Centralizar os dados (X - mu)
        X_centralizado = X - self.media_

        # Passo 3: Estimar a matriz de covariância
        # Nota: Usando rowvar=False porque X é (N_amostras, N_features)
        # Se você tiver sua função mcovar4, substitua aqui: CovX = mcovar4(X_centered.T)
        CovX = np.cov(X_centralizado, rowvar=False)

        # Passo 4: Decomposição SVD na Matriz de Covariância
        # Na covariância (simétrica), SVD e Eigendecomposition são equivalentes.
        # U: Autovetores (Matriz de Rotação)
        # S: Autovalores (Variâncias ao longo dos eixos)
        # Vt: Transposta de U
        U, S, Vt = np.linalg.svd(CovX)

        # S já vem ordenado decrescentemente no numpy.linalg.svd
        self.variancia_explicada_ = S
        self.razao_variancia_explicada_ = S / np.sum(S)

        # Passo 5: Determinar o número de componentes (q)
        if self.n_componentes is None:
            n_componentes_selec = X.shape[1]
        elif isinstance(self.n_componentes, float) and 0 < self.n_componentes < 1:
            # Seleciona q tal que a variância acumulada atinja o limiar (ex: 0.95)
            variancia_cum = np.cumsum(self.razao_variancia_explicada_)
            n_componentes_selec = np.searchsorted(variancia_cum, self.n_componentes) + 1
        else:
            n_componentes_selec = int(self.n_componentes)

        # Armazena apenas os q primeiros autovetores (colunas de U)
        # Estes formam a matriz de transformação W (p x q)
        self.componentes_ = U[:, :n_componentes_selec]

        return self

    def transform(self, X):
        """
        Projeta os dados originais no novo espaço de características reduzido.
        Z = (X - mu) @ W
        """
        if self.media_ is None or self.componentes_ is None:
            raise RuntimeError("O modelo PCA não foi treinado (fit).")

        # 1. Centralizar os dados de teste usando a MÉDIA DO TREINO
        X_centralizado = X - self.media_

        # 2. Projeção Linear
        # (N, p) @ (p, q) -> (N, q)
        return X_centralizado @ self.componentes_

    def inverse_transform(self, Z):
        """
        Reconstrói os dados para o espaço original (com perda de informação).
        X_rec = Z @ W.T + mu
        """
        # (N, q) @ (q, p) -> (N, p)
        return (Z @ self.componentes_.T) + self.media_  # type: ignore
