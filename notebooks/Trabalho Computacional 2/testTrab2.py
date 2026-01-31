# %% [markdown]
# # Experimentos do Trabalho 2 - Reconhecimento de Padrões
# - Aluno: Lucas José Lemos Braz

# %% [markdown]
# ### 1. Imports e Configurações

# %%
import time
import warnings
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Adicionando o diretório raiz ao path para importar os modelos
sys.path.append(os.path.abspath("../../.."))
from classificao_padroes.models_trabalho2 import mcovar1, mcovar2, mcovar3, mcovar4

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# %% [markdown]
# ### 2. Carregamento dos Dados
# Nesta seção, carregamos os datasets do robô Wall-Following.
# - `sensor_readings_24.data`: 24 sensores de ultrassom.
# - `sensor_readings_4.data`: 4 sensores de ultrassom simplificados.

# %%
# Caminhos relativos para os arquivos de dados
data_path_24 = "../../data/interin/wall+following+robot+navigation+data/sensor_readings_24.data"
data_path_4 = "../../data/interin/wall+following+robot+navigation+data/sensor_readings_4.data"

# Nomes das colunas
col_names_24 = [f"US{i+1}" for i in range(24)] + ["Class"]
col_names_4 = ["SD_front", "SD_left", "SD_right", "SD_back", "Class"]

# Carregando os datasets
df_24 = pd.read_csv(data_path_24, names=col_names_24)
df_4 = pd.read_csv(data_path_4, names=col_names_4)

# Separando features (X)
X_24 = df_24.drop("Class", axis=1).values
X_4 = df_4.drop("Class", axis=1).values

print(f"Dataset 24 sensores: {df_24.shape}")
print(f"Dataset 4 sensores: {df_4.shape}")

# %% [markdown]
# ### 3. Funções Auxiliares
# Definimos as funções para cálculo de covariância, norma e utilitários para plotagem e benchmarking.
# Agora incluímos também os métodos `mcovar1`, `mcovar2` e `mcovar4`.

# %%
def calc_cov_matrix(data):
    """Calcula a matriz de covariância (variáveis nas colunas) usando NumPy."""
    return np.cov(data, rowvar=False)

def calc_norm_diff(cov_np: np.ndarray, data: np.ndarray):
    """Calcula a norma entre o a matriz de covariância e a matriz de covariância calculada pelo numpy."""
    return np.linalg.norm(cov_np - data)

def plot_covariance_heatmap(data, col_labels, title):
    """Plota o heatmap da matriz de covariância."""
    cov_matrix = calc_cov_matrix(data)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".1f", square=True,cbar=False,
                xticklabels=col_labels, yticklabels=col_labels)
    plt.title(title)
    plt.show()

def run_benchmark(data, n_rounds=100):
    """Executa o benchmark e retorna listas de tempos."""
    methods = ['mcovar1', 'mcovar2', 'mcovar3', 'mcovar4', 'NumPy Cov']
    results = {m: [] for m in methods}
    
    # Dados transpostos para os métodos mcovar (esperam (p, N))
    data_T = data.T
    
    for _ in range(n_rounds):
        
        # mcovar1 (Loop)
        start = time.perf_counter()
        mcovar1(data_T)
        results['mcovar1'].append(time.perf_counter() - start)

        # mcovar2 (Matrix Ops)
        start = time.perf_counter()
        mcovar2(data_T)
        results['mcovar2'].append(time.perf_counter() - start)

        # mcovar3 (Outer Product)
        start = time.perf_counter()
        mcovar3(data_T)
        results['mcovar3'].append(time.perf_counter() - start)
        
        # mcovar4 (Outer Product)
        start = time.perf_counter()
        mcovar4(data_T)
        results['mcovar4'].append(time.perf_counter() - start)

        # NumPy Covariance
        start = time.perf_counter()
        calc_cov_matrix(data)
        results['NumPy Cov'].append(time.perf_counter() - start)
        
    return results

# %% [markdown]
# ### 4. Análise da Matriz de Covariância
# Visualização da matriz de covariância para ambos os casos.

# %%
# Visualização da Matriz de Covariância para 24 Sensores
plot_covariance_heatmap(X_24, col_names_24[:-1], "Matriz de Covariância (24 Sensores)")

# Visualização da Matriz de Covariância para 4 Sensores
plot_covariance_heatmap(X_4, col_names_4[:-1], "Matriz de Covariância (4 Sensores)")

# %% [markdown]
# ### Calculando norma das matrizes de covariância

# %%
# Calculando a norma entre a matriz de covariancia e a matriz de covariancia de referencia para cada metodo.
mcovars = [mcovar1, mcovar2, mcovar3, mcovar4]
C_ref_4 = np.cov(X_4.T)
C_ref_24 = np.cov(X_24.T)
normas_4 = []
normas_24 = []
print("Norma p/ 4 sensores.")
for i in range(4):
    C = mcovars[i](X_4.T)
    norma = np.linalg.norm(C - C_ref_4)
    normas_4.append(norma)
    print(f"Mcovar {i+1} | norma(C - C_ref): {norma}")

print("-" * 50)
print("Norma p/ 24 sensores.")
for i in range(4):
    C = mcovars[i](X_24.T)
    norma = np.linalg.norm(C - C_ref_24)
    normas_24.append(norma)
    print(f"Mcovar {i+1} | norma(C - C_ref): {norma}")


# %% [markdown]
# ### 5. Comparativo de Desempenho (Lado a Lado)
# Executamos o benchmark para ambos os datasets, agora incluindo os métodos externos `mcovar1`, `mcovar2`, e `mcovar4`.
# Note que `mcovar1` utiliza um loop explícito e pode ser significativamente mais lento.

# %%
# 1. Executar Benchmarks
print("Executando benchmark para 24 sensores (100 rodadas)...")
results_24 = run_benchmark(X_24)

print("Executando benchmark para 4 sensores (100 rodadas)...")
results_4 = run_benchmark(X_4)

# 2. Preparar DataFrames
def prepare_df(results):
    data_list = []
    for method, times in results.items():
        for t in times:
            data_list.append({'Tempo (s)': t, 'Método': method})
    return pd.DataFrame(data_list)

df_24 = prepare_df(results_24)
df_4 = prepare_df(results_4)

# 3. Plotagem Lado a Lado (2x2 Grid)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Paleta de Cores (distintas para cada método)
color_palette = {
    'mcovar1': 'blue',
    'mcovar2': 'green',
    'mcovar3': 'red',
    'mcovar4': 'purple',
    'NumPy Cov': 'orange'
}

# Ajuste de escala logarítmica para os gráficos de violino pode ser necessária se mcovar1 for muito lento
# Mas por padrão mantemos linear e observamos a discrepância.

# Histograma - 24 Sensores
sns.histplot(data=df_24, x='Tempo (s)', hue='Método', kde=True, element="step", ax=axes[0, 0], palette=color_palette, log_scale=True, stat="density")
axes[0, 0].set_title('Histograma - 24 Sensores')

# Histograma - 4 Sensores
sns.histplot(data=df_4, x='Tempo (s)', hue='Método', kde=True, element="step", ax=axes[0, 1], palette=color_palette, log_scale=True, stat="density")
axes[0, 1].set_title('Histograma - 4 Sensores')

# Violin Plot - 24 Sensores
sns.violinplot(data=df_24, x='Método', y='Tempo (s)', inner="quartile", ax=axes[1, 0], palette=color_palette, legend=True, log_scale=True)
axes[1, 0].set_title('Violin Plot - 24 Sensores')

# Violin Plot - 4 Sensores
sns.violinplot(data=df_4, x='Método', y='Tempo (s)', inner="quartile", ax=axes[1, 1], palette=color_palette, legend=True, log_scale=True)
axes[1, 1].set_title('Violin Plot - 4 Sensores')

plt.tight_layout()
plt.show()

# 4. Estatísticas
def print_stats(results, title):
    print(f"\n=== {title} ===")
    for method in results.keys():
        mean_time = np.mean(results[method])
        std_time = np.std(results[method])
        print(f"{method:<12} - Média: {mean_time:.6f} s | Desvio: {std_time:.6f} s")

print_stats(results_24, "Resultados Estatísticos (24 Sensores)")
print_stats(results_4, "Resultados Estatísticos (4 Sensores)")

# %% [markdown]
# ### Testando estatisticas sobre os resultados
# - Agora testaremos se o calculo de covariancia do numpy e os metodos implementados, entre si, são estatisticamente diferentes, estamos interessados especialmente se `mcovar1` e `mcovar3` são estatisticamente diferentes. Assim como `mcovar4`e `NumPy Cov`.

# %%
from scipy.stats import mannwhitneyu
df_24_m1 = df_24[df_24['Método'] == 'mcovar1']
df_24_m2 = df_24[df_24['Método'] == 'mcovar2']
df_24_m3 = df_24[df_24['Método'] == 'mcovar3']
df_24_m4 = df_24[df_24['Método'] == 'mcovar4']
df_24_m5 = df_24[df_24['Método'] == 'NumPy Cov']

# %%
stats, p = mannwhitneyu(df_24_m5['Tempo (s)'], df_24_m4['Tempo (s)'])
if p < 0.05:
    print("Existem diferenças significativas entre os métodos mcovar4 e Numpy Cov")
else:
    print("Não existem diferenças significativas entre os métodos")

# %%
stats, p = mannwhitneyu(df_24_m2['Tempo (s)'], df_24_m4['Tempo (s)'])
if p < 0.05:
    print("Existem diferenças significativas entre os métodos mcovar2 e mcovar4")
else:
    print("Não existem diferenças significativas entre os métodos")

# %%
stats, p = mannwhitneyu(df_24_m1['Tempo (s)'], df_24_m3['Tempo (s)'])
if p < 0.05:
    print("Existem diferenças significativas entre os métodos mcovar1 e mcovar3")
else:
    print("Não existem diferenças significativas entre os métodos")


