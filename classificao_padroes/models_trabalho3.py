import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
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
    cov_matrices_ : dict
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
        self.cov_matrices_ = {}
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
            self.cov_matrices_[c] = mcovar4(X_c.T)

    def predict(self, X):
        scores = []
        for c in self.classes_:
            diff = X - self.medias_[c]
            cov = self.cov_matrices_[c]

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


class ClassificadorDMP:
    """
    Classificador de Distância Mínima ao Protótipo (DMP) com Seleção Automática de K.
    Autor: Adaptação para RecPad-PPGETI
    """

    def __init__(self, k_min=2, k_max=10, n_runs=20):
        """
        Args:
            k_min (int): Mínimo de clusters a testar (>=2 para índices de validação).
            k_max (int): Máximo de clusters a testar.
            n_runs (int): Número de reinicializações do K-Means (Passo 2/3).
        """
        self.k_min = max(2, k_min)  # Indices precisam de pelo menos 2 clusters
        self.k_max = k_max
        self.n_runs = n_runs
        self.all_prototypes_ = []
        self.all_labels_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.all_prototypes_ = []
        self.all_labels_ = []

        print(f"Treinando DMP Multi-Protótipo. Classes: {self.classes_}")

        # Passo 1: Separar os dados por classe
        for c in self.classes_:
            X_c = X[y == c]
            n_samples = X_c.shape[0]

            # Proteção: Não podemos ter mais clusters que amostras
            # E precisamos de pelo menos k_min amostras
            limit_k = min(self.k_max, n_samples - 1)

            if limit_k < self.k_min:
                print(f"  [Aviso] Classe {c}: Amostras insuficientes ({n_samples}) para validação de clusters.")
                print("          Usando média simples (1 protótipo).")
                self.all_prototypes_.append(np.mean(X_c, axis=0))
                self.all_labels_.append(c)
                continue

            # Dicionário para "cachear" os modelos vencedores de cada K
            # Estrutura: {k: {'model': KMeansObject, 'db': float, 'ch': float, 'dn': float}}
            candidates = {}

            # Loop para variar K
            for k in range(self.k_min, limit_k + 1):
                best_inertia = np.inf
                best_model_k = None

                # Passo 2 e 3: Rodar K-Means Nr vezes e escolher o de menor inércia (SSD)
                # Isso garante que estamos avaliando o "melhor" K-Means possível para este k
                for _ in range(self.n_runs):
                    kmeans = KMeans(n_clusters=k, n_init=1, init='k-means++')
                    kmeans.fit(X_c)

                    if kmeans.inertia_ < best_inertia:
                        best_inertia = kmeans.inertia_
                        best_model_k = kmeans  # Guarda o objeto treinado

                # Passo 4: Calcular índices de validação para o modelo vencedor deste K
                labels = best_model_k.labels_  # type: ignore

                # Armazena tudo no dicionário de candidatos
                candidates[k] = {
                    'model': best_model_k,  # O IMPORTANTE ESTÁ AQUI: Guardamos o modelo!
                    'db': davies_bouldin_score(X_c, labels),      # Menor é melhor
                    'ch': calinski_harabasz_score(X_c, labels),   # Maior é melhor
                    'dunn': dunn_index(X_c, labels)               # Maior é melhor
                }

            # Passo 5: Votação para escolher K_opt
            # Extrair listas para buscar min/max
            ks_tested = list(candidates.keys())
            dbs = [candidates[k]['db'] for k in ks_tested]
            chs = [candidates[k]['ch'] for k in ks_tested]
            dns = [candidates[k]['dunn'] for k in ks_tested]

            # Votos
            k_best_db = ks_tested[np.argmin(dbs)]
            k_best_ch = ks_tested[np.argmax(chs)]
            k_best_dn = ks_tested[np.argmax(dns)]

            votes = [k_best_db, k_best_ch, k_best_dn]

            # Lógica de Votação e Desempate
            vote_counts = Counter(votes)
            most_common = vote_counts.most_common()  # Retorna lista [(k, count), ...]

            # Se houver um vencedor claro (count > 1) ou empate, pegamos o primeiro.
            # Se houver empate total (1, 1, 1), precisamos de um critério.
            # Critério: Preferência pelo índice DB (Davies-Bouldin)
            if vote_counts[k_best_db] == 1 and vote_counts[k_best_ch] == 1 and vote_counts[k_best_dn] == 1:
                k_opt = k_best_db
                reason = "Desempate via DB"
            else:
                k_opt = most_common[0][0]
                reason = "Maioria"

            print(f"  Classe {c}: Votos [DB={k_best_db}, CH={k_best_ch}, Dunn={k_best_dn}] -> Escolhido: {k_opt} ({reason})")

            # Passo 6 (Corrigido): Recuperar os protótipos do modelo JÁ TREINADO
            # Não fazemos fit() novamente aqui.
            winner_model = candidates[k_opt]['model']

            for centroide in winner_model.cluster_centers_:
                self.all_prototypes_.append(centroide)
                self.all_labels_.append(c)

        # Converter para numpy array para eficiência no predict
        self.all_prototypes_ = np.array(self.all_prototypes_)
        self.all_labels_ = np.array(self.all_labels_)

    def predict(self, X):
        """
        Classifica cada amostra em X com base no protótipo mais próximo (Euclidiana).
        """
        if len(self.all_prototypes_) == 0:
            raise ValueError("O modelo não foi treinado ou não gerou protótipos.")

        # Calcula distância de todas as amostras para todos os protótipos (Matriz N x N_prototypes)
        dists = cdist(X, self.all_prototypes_, metric='euclidean')

        # Índice do protótipo com menor distância
        nearest_proto_idx = np.argmin(dists, axis=1)

        # Recupera o rótulo da classe associado àquele protótipo
        return self.all_labels_[nearest_proto_idx]


class PCA:
    """
    Principal Component Analysis (PCA) via SVD da Matriz de Covariância.
    """

    def __init__(self, n_components=None):
        """
        n_components:
            - Se float (0 < n < 1): Seleciona componentes para explicar 'n' % da variância.
            - Se int (n >= 1): Seleciona exatamente 'n' componentes.
            - Se None: Mantém todos os componentes.
        """
        self.n_components = n_components
        self.components_ = None      # Matriz de projeção (W)
        self.mean_ = None            # Média para centralização
        self.explained_variance_ = None       # Autovalores (lambda)
        self.explained_variance_ratio_ = None  # Variância explicada (%)

    def fit(self, X):
        # Passo 1: Calcular a média de cada atributo
        self.mean_ = np.mean(X, axis=0)

        # Passo 2: Centralizar os dados (X - mu)
        X_centered = X - self.mean_

        # Passo 3: Estimar a matriz de covariância
        # Nota: Usando rowvar=False porque X é (N_amostras, N_features)
        # Se você tiver sua função mcovar4, substitua aqui: CovX = mcovar4(X_centered.T)
        CovX = np.cov(X_centered, rowvar=False)

        # Passo 4: Decomposição SVD na Matriz de Covariância
        # Na covariância (simétrica), SVD e Eigendecomposition são equivalentes.
        # U: Autovetores (Matriz de Rotação)
        # S: Autovalores (Variâncias ao longo dos eixos)
        # Vt: Transposta de U
        U, S, Vt = np.linalg.svd(CovX)

        # S já vem ordenado decrescentemente no numpy.linalg.svd
        self.explained_variance_ = S
        self.explained_variance_ratio_ = S / np.sum(S)

        # Passo 5: Determinar o número de componentes (q)
        if self.n_components is None:
            n_components_select = X.shape[1]
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Seleciona q tal que a variância acumulada atinja o limiar (ex: 0.95)
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)
            n_components_select = np.searchsorted(cumulative_variance, self.n_components) + 1
        else:
            n_components_select = int(self.n_components)

        # Armazena apenas os q primeiros autovetores (colunas de U)
        # Estes formam a matriz de transformação W (p x q)
        self.components_ = U[:, :n_components_select]

        return self

    def transform(self, X):
        """
        Projeta os dados originais no novo espaço de características reduzido.
        Z = (X - mu) @ W
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("O modelo PCA não foi treinado (fit).")

        # 1. Centralizar os dados de teste usando a MÉDIA DO TREINO
        X_centered = X - self.mean_

        # 2. Projeção Linear
        # (N, p) @ (p, q) -> (N, q)
        return X_centered @ self.components_

    def inverse_transform(self, Z):
        """
        Reconstrói os dados para o espaço original (com perda de informação).
        X_rec = Z @ W.T + mu
        """
        # (N, q) @ (q, p) -> (N, p)
        return (Z @ self.components_.T) + self.mean_  # type: ignore
