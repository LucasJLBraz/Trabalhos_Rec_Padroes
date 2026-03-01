# Gemini Context: Reconhecimento de Padrões - PPGETI

This file provides context and instructions for AI agents working on the "Trabalhos de Reconhecimento de Padrões" project.

## Project Overview

This repository contains the implementation of algorithms, models, and experiments for three Computational Tasks (TC) of the Pattern Recognition course at PPGETI (Programa de Pós-Graduação em Engenharia de Teleinformática).

- **Student:** Lucas José Lemos Braz
- **Core Technologies:** Python 3.12, `uv` (dependency management), NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebooks.
- **Project Structure:**
    - `classificao_padroes/`: Main Python module containing core logic and models.
        - `models_trabalho1.py`: Models for TC1 (KNN, DMC, etc.).
        - `models_trabalho2.py`: Models and functions for TC2 (Covariance matrices, etc.).
        - `models_trabalho3.py`: Models and functions for TC3 (Clustering, etc.).
    - `notebooks/`: Jupyter notebooks for each task's experiments.
    - `data/`: Raw and interim datasets.
    - `docs/`: LaTeX reports and MkDocs documentation.
    - `references/`: Course materials, task descriptions (PDFs), and support files.
    - `reports/`: Generated figures and technical reports.

## Relatórios em LaTeX: Padrão de Excelência (Diretrizes para o Agente)

Para construir os relatórios (TC1, TC2, TC3), o Agente deve seguir o protocolo de "Análise Profunda" baseado no modelo de referência de Lucas Braz.

- Utilize os resultados e discussões dos notebooks de cada trabalho para construir o relatorio.


### 1. Protocolo de Interação com o Usuário
O Agente não deve gerar o texto final sem antes:
- **Solicitar Figuras e Tabelas:** O Agente deve pedir explicitamente: "Por favor, forneça as figuras de [matriz de confusão/histogramas/violin plots] e as tabelas de resultados geradas no notebook." 
- O agente pode gerar o LaTeX sem as figuras, deixando claro o que deve ser inserido.
- **Solicitar Exportação de Código:** O Agente deve instruir o usuário a gerar versões PDF ou Markdown dos notebooks via Pandoc (`nbconvert`) para revisar a implementação lógica antes de descrevê-la.
- **Trabalho por Etapas:** Cada TC deve ser tratado como um projeto isolado, garantindo que o escopo não se misture.

### 2. Estrutura Obrigatória do Relatório (LaTeX)

- UTILIZE O PROJETO EM LATEX NA PASTA `references/Modelo Trabalho Latex/Modelo___Trabalho_1_ICAp___SBC` COMO PADRÃO OURO DE ANALISE.

- Evite o uso excessivo de subseções e bullet points. Tente deixar o texto mais fluido, conectando uma seção com a outra.

Cada relatório deve conter, no mínimo:
- **Introdução:** Contextualização do problema, definição da tarefa (classificação vs. estimação), e objetivos claros.
- **Metodologia:** - Descrição matemática dos modelos (ex: Definição da distância de Minkowski para o KNN: $d(x, y) = (\sum |x_i - y_i|^m)^{1/m}$).
    - Protocolo experimental (N-rodadas, divisão treino/teste).
    - Justificativa das decisões de modelagem (ex: Por que usar DMC robusto? Por que regularizar a matriz de covariância?).
- **Resultados e Discussão Crítica:** - Não apenas listar acurácia. Discutir o *trade-off*
    - Análise de tempos de execução (complexidade computacional prática).
- **Conclusão:** Síntese das lições aprendidas e limitações do modelo.

### 3. Análises Específicas por Trabalho (Escopo Crítico)

#### TC1: Classificação (Vertebral Column)
- **Influência da Métrica:** Analisar como a ordem $m$ da distância de Minkowski afeta a fronteira de decisão. 
- **Estabilidade:** Discutir a variação da acurácia entre as 100 rodadas (desvio-padrão como medida de robustez).
- **Análise de Erros:** Usar as matrizes de confusão da melhor e pior rodada para identificar quais classes são mais sobrepostas (pathologies vs. normal).

#### TC2: Matrizes de Covariância (Wall Following Robot)
- **Eficiência vs. Precisão:** Comparar os métodos manuais (1 a 4) com a função nativa em termos de erro residual ($E=C_{my}-C_{ref}$) e tempo.
- **Condicionamento Numérico:** Discutir o número de condicionamento ($rcond$) e o posto da matriz. Se for mal-condicionada, justificar a técnica de regularização (ex: $C + \epsilon I$).

#### TC3: Modelos Probabilísticos e Redução de Dimensionalidade
- **PCA:** Não apenas plotar a variância. Determinar o "joelho" (elbow) do gráfico e justificar a escolha de $q$ componentes.
- **Impacto da Redução:** Comparar o desempenho do CQG e DMP com e sem PCA. Houve perda de informação relevante?
- Discutir porque em um modelo com baixissima correlação na covariancia, causa um PCA menos efetivo na redução de dimensões
- **Protótipos:** Explicar como a escolha do número de protótipos via K-médias influencia a capacidade de generalização do DMP.

### 4. Convenções de Repositório
Todos os relatórios devem referenciar a organização do repositório oficial:
`https://github.com/LucasJLBraz/Trabalhos_Rec_Padroes`
O texto deve explicar que a modularização (separação entre `models_trabalhoX.py` e `notebooks/`) garante a reprodutibilidade científica.

### 5. Tom e Estilo
- Usar terminologia técnica precisa.
- Manter o rigor matemático: definir dimensões de matrizes (ex: $X \in \mathbb{R}^{p \times N}$).
- Desafiar o usuário a pensar sobre "por que" um modelo falhou em determinado cenário.