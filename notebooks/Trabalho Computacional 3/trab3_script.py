# %% [markdown]
#  # Trabalho Computacional 3 - Reconhecimento de Padrões
# 
#  ## Classificadores Gaussianos, DMP e Análise de Componentes Principais (PCA)
# 
#  **Aluno:** Lucas José Lemos Braz
# 
#  **Disciplina:** Reconhecimento de Padrões (Pós-Graduação)

# %% [markdown]
#  ### 1. Configurações e Importações
# 
#  Importação das bibliotecas científicas e configuração do ambiente de visualização.

# %%
import sys
import os
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ferramentas do Scikit-Learn para validação e métricas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Adicionando o diretório raiz ao path para importar os modelos customizados
sys.path.append(os.path.abspath("../../.."))

# IMPORTANTE: Assume-se que este arquivo contém as implementações corrigidas
# (DMP com seleção de K via DB/CH/Dunn e CQG com regularização/SVD)
from classificao_padroes.models_trabalho3 import (
    analizar_invertibilidade,
    PCA,
    ClassificadorDMP,
    ClassificadorQuadraticoGaussiano
)

# Configurações de plotagem
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)


# %% [markdown]
#  ### 2. Carregamento e Pré-processamento dos Dados
# 
#  O dataset *Wall-Following Robot* possui 24 sensores de ultrassom dispostos circularmente.
# 
#  Os dados são carregados e normalizados (Z-score) para garantir estabilidade numérica, especialmente para o DMP (baseado em distância Euclidiana).

# %%
# Caminhos e carregamento
data_path_24 = "../../data/interin/wall+following+robot+navigation+data/sensor_readings_24.data"
col_names_24 = [f"US{i + 1}" for i in range(24)] + ["Class"]

data_df = pd.read_csv(data_path_24, names=col_names_24)

# Separação de Atributos e Classes
X_raw = data_df.drop("Class", axis=1).values
y_labels = data_df["Class"].values

# Codificação das classes (String -> Inteiro) para facilitar métricas
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
classes_nomes = le.classes_



# Normalização Z-Score (Média 0, Desvio 1)
# Crítico para classificadores baseados em distância (DMP) e covariância (CQG)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_raw)
X_scaled = X_raw  # Mantendo os dados originais para análise de invertibilidade e PCA

print(f"Dataset Carregado: {X_scaled.shape}")
print(f"Classes: {classes_nomes}")

# Verificação de Balanceamento
plt.figure(figsize=(8, 4))
sns.countplot(x=y_labels, palette='viridis', order=classes_nomes)
plt.title("Distribuição das Classes")
plt.xlabel("Classes")
plt.ylabel("Número de Amostras")
plt.show()


# %% [markdown]
#  ### 3. Funções Auxiliares de Avaliação Estatística
# 
#  Como exigido em trabalhos de pós-graduação, não avaliamos o modelo em apenas uma rodada.
# 
#  Esta função executa **Nr=100 rodadas** de Monte Carlo (divisões aleatórias de treino/teste) para gerar estatísticas robustas (Média e Desvio Padrão).

# %%
def executar_experimento_monte_carlo(X, y, classificador_class, params={}, n_runs=20, test_size=0.3, q_pca=None):
    """
    Executa N rodadas de treino e teste e coleta métricas globais e por classe.
    """
    print(f"Iniciando {n_runs} rodadas para {classificador_class.__name__}...")

    global_accuracies = []
    class_accuracies = {c: [] for c in np.unique(y)}
    times_train = []
    times_test = []

    for i in range(n_runs):
        # Divisão aleatória
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if q_pca is not None:
            pca = PCA(n_components=q_pca)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

        # Instanciação e Treino
        clf = classificador_class(**params)

        start_train = time.time()
        clf.fit(X_train, y_train)
        times_train.append(time.time() - start_train)

        # Teste
        start_test = time.time()
        y_pred = clf.predict(X_test)
        times_test.append(time.time() - start_test)

        # Métricas Globais
        global_accuracies.append(accuracy_score(y_test, y_pred))

        # Métricas por Classe (Recall/Acurácia por classe)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        for cls_label in np.unique(y):
            # O report usa strings dos labels se y for string, ou str(int) se y for int
            cls_key = str(cls_label)
            if cls_key in report:
                class_accuracies[cls_label].append(report[cls_key]['recall'])
            else:
                class_accuracies[cls_label].append(0.0)

    # Compilação dos Resultados
    results = {
        'Modelo': classificador_class.__name__,
        'Global_Mean': np.mean(global_accuracies),
        'Global_Std': np.std(global_accuracies),
        'Time_Train_Mean': np.mean(times_train),
        'Time_Test_Mean': np.mean(times_test)
    }

    # Adiciona média e desvio por classe
    for cls_label in np.unique(y):
        results[f'Class_{cls_label}_Mean'] = np.mean(class_accuracies[cls_label])
        results[f'Class_{cls_label}_Std'] = np.std(class_accuracies[cls_label])

    return results, global_accuracies


def formatar_tabela_latex(df_results, class_names):
    """Gera uma visualização amigável similar à tabela solicitada no PDF."""
    display_df = pd.DataFrame()
    display_df['Classificador'] = df_results['Modelo']

    # Formata Global
    display_df['Global'] = df_results.apply(lambda row: f"{row['Global_Mean']:.4f} ± {row['Global_Std']:.4f}", axis=1)

    # Formata por Classe
    for i, name in enumerate(class_names):
        display_df[f'{name}'] = df_results.apply(
            lambda row: f"{row[f'Class_{i}_Mean']:.4f} ± {row[f'Class_{i}_Std']:.4f}", axis=1
        )

    return display_df


# %% [markdown]
#  ---
# 
#  ## QUESTÃO 1: Análise sem PCA (Espaço Original 24D)
# 
# 
# 
#  ### 1.1 e 1.2: Análise de Invertibilidade
# 
#  Verifica-se se as matrizes de covariância são bem condicionadas para aplicação do CQG.

# %%
# A função analisar_invertibilidade deve ser capaz de lidar com DataFrame ou Array
# Criando DF temporário apenas para essa função, se necessário
analizar_invertibilidade(data_df, "Original 24 Sensores")


# %% [markdown]
#  ### 1.3: Experimento Comparativo (DMP vs CQG)
# 
#  Execução de 100 rodadas para preenchimento da tabela de desempenho.
# 
# 
# 
#  *Nota:* Para o DMP, definimos `k_max` baseado na heurística $\sqrt{N_{min}} \cdot 2$, para evitar overfitting (transformar o DMP em 1-NN).

# %%
# Configuração do DMP
N_min = data_df["Class"].value_counts().min()
k_max_heuristic = int(np.sqrt(N_min) * 2)  # Multiplicamos por 2 para permitir um pouco mais de flexibilidade, mas ainda evitar overfitting extremo
print(f"Heurística para DMP: k_max definido como {k_max_heuristic} (aprox sqrt({N_min}))")

# Parâmetros dos Modelos
params_dmp = {'k_min': 2, 'k_max': k_max_heuristic, 'n_runs': 1}  # n_runs interno do k-means
params_cqg = {}  # CQG padrão

# Execução (Pode levar alguns minutos devido às 100 rodadas)
res_dmp, acc_dmp = executar_experimento_monte_carlo(X_scaled, y_encoded, ClassificadorDMP, params_dmp, n_runs=20)
res_cqg, acc_cqg = executar_experimento_monte_carlo(X_scaled, y_encoded, ClassificadorQuadraticoGaussiano, params_cqg, n_runs=20)

# Consolidação
df_q1 = pd.DataFrame([res_cqg, res_dmp])

print("\n--- Tabela de Resultados (Questão 1.3) ---")
tabela_q1 = formatar_tabela_latex(df_q1, classes_nomes)
display(tabela_q1)

# Boxplot para comparação visual da estabilidade
plt.figure(figsize=(8, 5))
plt.boxplot([acc_cqg, acc_dmp], labels=['CQG', 'DMP'])
plt.title("Distribuição de Acurácia Global (100 Rodadas)")
plt.ylabel("Acurácia")
plt.grid(True)
plt.show()


# %% [markdown]
#  ---
# 
#  ## QUESTÃO 2: Aplicação de PCA
# 
# 
# 
#  ### 2.1: Determinação do número de componentes (q)
# 
#  Analisamos o espectro de autovalores para escolher um `q` que reduza a dimensionalidade retendo pelo menos 95% da informação (variância).

# %%
# Ajuste do PCA na base completa (apenas para análise de variância)
pca_full = PCA(num_componentes=None)
pca_full.fit(X_scaled)

# Cálculo das variâncias
var_ratio = pca_full.razao_variancia_explicada_
var_cum = np.cumsum(var_ratio)

# Plotagem Profissional (Scree Plot + Acumulada)
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Número de Componentes (q)')
ax1.set_ylabel('Variância Explicada Individual (%)', color=color)
ax1.bar(range(1, len(var_ratio) + 1), var_ratio * 100, color=color, alpha=0.6, label='Individual')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Segundo eixo y
color = 'tab:red'
ax2.set_ylabel('Variância Acumulada (%)', color=color)
ax2.plot(range(1, len(var_cum)+1), var_cum * 100, color=color, marker='o', linewidth=2, label='Acumulada')
ax2.tick_params(axis='y', labelcolor=color)

# Linhas de corte (90%, 95%, 99%)
for threshold, style in zip([0.90, 0.95, 0.99], [':', '--', '-.']):
    q_idx = np.argmax(var_cum >= threshold) + 1
    ax2.axhline(y=threshold * 100, color='gray', linestyle=style, alpha=0.5)
    ax2.text(q_idx + 0.5, threshold * 100 - 2, f'{threshold * 100:.0f}% (q={q_idx})', color='black')
    print(f"Componentes necessários para {threshold * 100:.0f}% de variância: q = {q_idx}")

plt.title("Análise de Variância Explicada - PCA")
fig.tight_layout()
plt.show()

# Definição do q escolhido (Critério > 95%)
q_selected = np.argmax(var_cum >= 0.95) + 1
print(f"\n---> Valor de q escolhido para a Questão 2.2: {q_selected} componentes.")


# %% [markdown]
# ### 2.2 Experimento Comparativo com PCA e Análises de Trade-off
# Nesta etapa, não aceitamos cegamente a heurística de 95% da variância. 
# Avaliamos empiricamente o impacto do número de componentes (q) na acurácia e no tempo de execução.
#  Transformação dos dados para `q` dimensões e repetição das 100 rodadas de teste.
# 
#  **Hipótese:** Espera-se que o desempenho se mantenha próximo ao original, mas com menor custo computacional no CQG (inversão de matrizes menores) e possivelmente melhor generalização no DMP (remoção de ruído).

# %%
# ==============================================================================
# 1. Análise Empírica: Acurácia vs Custo Computacional para CQG e DMP (Varredura de q)
# ==============================================================================
from sklearn.metrics import confusion_matrix


print("Realizando varredura de 'q' para justificação científica (CQG e DMP)...")
print("Aviso: Esta etapa pode demorar alguns minutos devido ao treinamento do DMP.")

q_test_values = [2, 5, 8, 12, 15, 18, 21, 24]

# Armazenamento de Históricos
acc_cqg_history, time_cqg_history = [], []
acc_dmp_history, time_dmp_history = [], []

# Número reduzido de rodadas APENAS para desenhar a curva em tempo hábil
n_runs_sweep = 10

# Se o DMP estiver demorando muito na varredura, podemos simplificar seus parâmetros
# apenas para esta etapa de descoberta geométrica.
params_dmp_sweep = {'k_min': 2, 'k_max': k_max_heuristic}

for q_test in q_test_values:
    print(f" -> Avaliando q = {q_test}...")

    # Avaliação CQG
    res_cqg_temp, _ = executar_experimento_monte_carlo(
        X_scaled, y_encoded, ClassificadorQuadraticoGaussiano, params_cqg, n_runs=n_runs_sweep, q_pca=q_test
    )
    acc_cqg_history.append(res_cqg_temp['Global_Mean'])
    time_cqg_history.append(res_cqg_temp['Time_Test_Mean'])

    # Avaliação DMP
    res_dmp_temp, _ = executar_experimento_monte_carlo(
        X_scaled, y_encoded, ClassificadorDMP, params_dmp_sweep, n_runs=n_runs_sweep, q_pca=q_test
    )
    acc_dmp_history.append(res_dmp_temp['Global_Mean'])
    time_dmp_history.append(res_dmp_temp['Time_Test_Mean'])

# ==============================================================================
# Plotagem Dupla: Acurácia (Eixo Esquerdo) e Tempo (Eixo Direito)
# ==============================================================================
# Dois gráficos (um por modelo): Acurácia (eixo esquerdo) e Tempo (eixo direito)
q_selected_cqg = q_test_values[np.argmax(acc_cqg_history)]
q_selected_dmp = q_test_values[np.argmax(acc_dmp_history)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# ==============================================================================
# Gráfico 1: CQG
# ==============================================================================
ax1 = axes[0]
ax1.set_title("CQG: Acurácia vs Tempo", fontsize=13)
ax1.set_xlabel("Número de Componentes Principais (q)", fontweight="bold")
ax1.set_ylabel("Acurácia Média", fontweight="bold")
ax1.plot(q_test_values, acc_cqg_history, marker="o", color="#1f77b4", linewidth=2, label="Acurácia (CQG)")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.axvline(x=q_selected_cqg, color="gray", linestyle=":", linewidth=2, label=f"q selecionado ({q_selected_cqg})")

ax1b = ax1.twinx()
ax1b.set_ylabel("Tempo Médio de Predição (s)", fontweight="bold")
ax1b.plot(q_test_values, time_cqg_history, marker="^", linestyle="--", color="#d62728", linewidth=2, label="Tempo (CQG)")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="best")

# ==============================================================================
# Gráfico 2: DMP
# ==============================================================================
ax2 = axes[1]
ax2.set_title("DMP: Acurácia vs Tempo", fontsize=13)
ax2.set_xlabel("Número de Componentes Principais (q)", fontweight="bold")
ax2.set_ylabel("Acurácia Média", fontweight="bold")
ax2.plot(q_test_values, acc_dmp_history, marker="s", color="#2ca02c", linewidth=2, label="Acurácia (DMP)")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.axvline(x=q_selected_dmp, color="gray", linestyle=":", linewidth=2, label=f"q selecionado ({q_selected_dmp})")

ax2b = ax2.twinx()
ax2b.set_ylabel("Tempo Médio de Predição (s)", fontweight="bold")
ax2b.plot(q_test_values, time_dmp_history, marker="D", linestyle="--", color="#ff7f0e", linewidth=2, label="Tempo (DMP)")

lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="best")

plt.tight_layout()
plt.show()

# ==============================================================================
# O restante do código de consolidação (Tabelas e Matriz de confusão) vem aqui...
# ==============================================================================
# ==============================================================================
# 2. Execução Oficial no Espaço Reduzido (q selecionado)
# ==============================================================================
pca_final_cqg = PCA(num_componentes=q_selected_cqg)
pca_final_cqg.fit(X_scaled)
X_pca_cqg = pca_final_cqg.transform(X_scaled)

print(f"\nTransformação Oficial: {X_scaled.shape} -> {X_pca_cqg.shape}")

# Execução do Experimento no Espaço Reduzido (20 rodadas para rigor estatístico)
res_cqg_pca, acc_cqg_pca = executar_experimento_monte_carlo(X_pca_cqg, y_encoded, ClassificadorQuadraticoGaussiano, params_cqg, n_runs=20)


pca_final_dmp = PCA(num_componentes=q_selected_dmp)
pca_final_dmp.fit(X_scaled)
X_pca_dmp = pca_final_dmp.transform(X_scaled)
res_dmp_pca, acc_dmp_pca = executar_experimento_monte_carlo(X_pca_dmp, y_encoded, ClassificadorDMP, params_dmp, n_runs=20)
# Consolidação da Tabela
df_q2 = pd.DataFrame([res_cqg_pca, res_dmp_pca])
df_q2['Modelo'] = df_q2['Modelo'] + " (PCA)"

print("\n--- Tabela de Resultados (Questão 2.2 - Com PCA) ---")
tabela_q2 = formatar_tabela_latex(df_q2, classes_nomes)
display(tabela_q2)

# ==============================================================================
# 3. Análise de Desbalanceamento (Matriz de Confusão Normalizada)
# ==============================================================================
# Vamos fazer um split único para gerar a matriz de confusão visual do DMP com PCA
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca_dmp, y_encoded, test_size=0.3, random_state=42)

dmp_viz = ClassificadorDMP(**params_dmp)
dmp_viz.fit(X_train_pca, y_train_pca)
y_pred_viz = dmp_viz.predict(X_test_pca)

cm = confusion_matrix(y_test_pca, y_pred_viz, normalize='true') # Normalizado por linha (Recall)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes_nomes, yticklabels=classes_nomes)
plt.title(f"Matriz de Confusão Normalizada - DMP com PCA (q={q_selected_dmp})\nAtenção ao viés na classe minoritária")
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Predita')
plt.show()

# ==============================================================================
# 4. Boxplots Comparativos (Acurácia e Tempo)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot 1: Acurácia
data_boxplot_acc = [acc_cqg, acc_cqg_pca, acc_dmp, acc_dmp_pca]
labels_boxplot = ['CQG', f'CQG+PCA\n(q={q_selected})', 'DMP', f'DMP+PCA\n(q={q_selected})']
sns.boxplot(data=data_boxplot_acc, ax=axes[0], palette='Set2')
axes[0].set_xticklabels(labels_boxplot)
axes[0].set_title("Comparativo de Acurácia Global")
axes[0].set_ylabel("Acurácia")

# Boxplot 2: Tempos de Treino/Teste baseados no histórico
# Extraindo os tempos diretamente do Dicionário de resultados para mostrar a diferença
tempos_execucao = pd.DataFrame({
    'Modelo': ['CQG', 'CQG+PCA', 'DMP', 'DMP+PCA'],
    'Tempo Teste (s)': [res_cqg['Time_Test_Mean'], res_cqg_pca['Time_Test_Mean'], 
                        res_dmp['Time_Test_Mean'], res_dmp_pca['Time_Test_Mean']]
})

sns.barplot(data=tempos_execucao, x='Modelo', y='Tempo Teste (s)', ax=axes[1], palette='Set2')
axes[1].set_title("Custo Computacional: Tempo Médio de Predição")
axes[1].set_ylabel("Segundos")

plt.tight_layout()
plt.show()

# ==============================================================================
# 5. Reflexão sobre a Topologia do DMP
# ==============================================================================
# Analisando quantos protótipos foram alocados para cada classe
contagem_prototipos = pd.Series(dmp_viz.labels_).map(dict(enumerate(classes_nomes))).value_counts()
print("\n--- Estrutura Topológica do DMP (Distribuição de Protótipos) ---")
print(contagem_prototipos)
print("\nComentário: Observe como o número de protótipos se correlaciona com o número de amostras da classe.")
print("Classes maiores exigem partições de Voronoi mais complexas.")

# %% [markdown]
#  ### Visualização Extra: Superfícies de Decisão (2D)
# 
#  Para fins didáticos, projetamos os dados nas 2 primeiras componentes principais e treinamos os modelos **apenas nestas 2 dimensões** para visualizar as regiões de decisão.
# 
#  *Nota: A performance aqui será inferior à de 24D ou qD, pois estamos usando apenas 2 dimensões.*

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_decision_boundary(X, y, model, title, le):
    # Treina modelo em 2D
    model.fit(X, y)

    # Meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predição no Grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    
    # Atualizado para as versões mais recentes do Matplotlib
    try:
        cmap = plt.cm.get_cmap('tab10')
    except AttributeError:
        import matplotlib as mpl
        cmap = mpl.colormaps['tab10']

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # ---------------------------------------------------------
    # FIX: Converte os números de volta para os nomes das classes
    # ---------------------------------------------------------
    y_class_names = le.inverse_transform(y)
    
    # Passamos 'y_class_names' para o hue em vez de 'y'
    sns.scatterplot(
        x=X[:, 0], 
        y=X[:, 1], 
        hue=y_class_names, 
        palette='tab10', 
        legend='full', 
        s=30, 
        alpha=0.6
    )

    # Se for DMP, plotar protótipos
    if hasattr(model, 'all_prototypes_') and len(model.all_prototypes_) > 0:
        prototypes = model.all_prototypes_
        prototype_labels = model.all_labels_
        
        # Opcional: decodificar os labels dos protótipos se quiser usá-los no hover ou logs
        prototype_labels_str = le.inverse_transform(prototype_labels)
        
        # Normalize integer indices for colormap (0 to num_classes-1)
        # Prevenindo erro de divisão por zero caso haja apenas a classe 0
        max_label = np.max(prototype_labels) if np.max(prototype_labels) > 0 else 1
        colors = cmap(prototype_labels / max_label)
        
        plt.scatter(prototypes[:, 0], prototypes[:, 1], 
                    c=colors, marker='X', s=100, label='Protótipos', edgecolors='white', linewidths=2)

    plt.title(title)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.show()

# --- Execução ---

# Preparação 2D
pca_2d = PCA(n_components=2)
pca_2d.fit(X_scaled)
X_2d = pca_2d.transform(X_scaled)

# Assumindo que 'le' é o seu LabelEncoder instanciado e "fitado" anteriormente
# ex: le = LabelEncoder()
# ex: y_encoded = le.fit_transform(y_strings)

# Visualização DMP 2D (Adicionando o argumento le=le)
dmp_2d = ClassificadorDMP(k_min=2, k_max=15, n_runs=10)
plot_decision_boundary(X_2d, y_encoded, dmp_2d, "Superfície de Decisão DMP (Projeção 2D)", le=le)

# Visualização CQG 2D (Adicionando o argumento le=le)
cqg_2d = ClassificadorQuadraticoGaussiano()
plot_decision_boundary(X_2d, y_encoded, cqg_2d, "Superfície de Decisão CQG (Projeção 2D)", le=le)

# %%
dmp_2d.labels_
le.inverse_transform(dmp_2d.labels_)

# %%



