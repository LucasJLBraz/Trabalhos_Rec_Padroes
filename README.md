# Reconhecimento de Padrões - PPGETI

Este repositório contém a implementação dos algoritmos, modelos e experimentos referentes aos três Trabalhos Computacionais (TC) da disciplina de Reconhecimento de Padrões do Programa de Pós-Graduação em Engenharia de Teleinformática (PPGETI).

## Descrição do Projeto

O objetivo deste projeto é consolidar as implementações de modelos de classificação, regressão e clusterização, bem como a análise de dados e geração de relatórios técnicos. Cada trabalho possui seu próprio conjunto de experimentos e documentação associada.

## Organização do Projeto

```text
├── LICENSE            <- Licença de código aberto (se aplicável).
├── Makefile           <- Makefile com comandos de conveniência como `make data`.
├── README.md          <- Este arquivo, contendo a visão geral do projeto.
├── pyproject.toml     <- Arquivo de configuração do projeto e dependências (gerenciado pelo uv).
├── uv.lock            <- Arquivo de trava de dependências do uv.
├── setup.cfg          <- Arquivo de configuração para ferramentas de linting.
│
├── classificao_padroes <- Código-fonte do projeto (módulo Python).
│   ├── modeling/      <- Scripts para treinamento e predição.
│   ├── config.py      <- Variáveis de configuração e caminhos.
│   ├── dataset.py     <- Scripts para download ou geração de dados.
│   ├── features.py    <- Código para engenharia de atributos.
│   ├── models_trabalho1.py <- Modelos específicos do Trabalho 1 (KNN, DMC, etc).
│   ├── models_trabalho2.py <- Modelos específicos do Trabalho 2.
│   ├── models_trabalho3.py <- Modelos específicos do Trabalho 3.
│   └── plots.py       <- Código para geração de visualizações.
│
├── data               <- Dados utilizados nos experimentos.
│   ├── interim/       <- Dados intermediários transformados.
│   └── raw/           <- Dados originais e imutáveis.
│
├── docs               <- Documentação e relatórios em LaTeX.
│   ├── relatorio_tc1/ <- Relatório em LaTeX do Trabalho 1.
│   ├── relatorio_tc2/ <- Relatório em LaTeX do Trabalho 2.
│   ├── relatorio_tc3/ <- Relatório em LaTeX do Trabalho 3.
│   └── mkdocs.yml     <- Configuração para documentação via MkDocs.
│
├── models             <- Modelos treinados e serializados ou sumários.
│
├── notebooks          <- Notebooks Jupyter com os experimentos práticos.
│   ├── Trabalho Computacional 1/ <- Experimentos do Trabalho 1.
│   ├── Trabalho Computacional 2/ <- Experimentos do Trabalho 2.
│   └── Trabalho Computacional 3/ <- Experimentos do Trabalho 3.
│
├── references         <- Enunciados dos trabalhos (TC1, TC2, TC3) e materiais de apoio.
│
└── reports            <- Análises geradas (HTML, PDF, etc.) e figuras.
    └── figures/       <- Gráficos e figuras geradas para os relatórios.
```

## Como Rodar os Experimentos

Este projeto utiliza o [uv](https://github.com/astral-sh/uv) para gerenciamento de pacotes e ambientes virtuais.

### Pré-requisitos

Certifique-se de ter o `uv` instalado em sua máquina. Caso não possua, siga as instruções na [documentação oficial](https://github.com/astral-sh/uv).

### Instalação

Para instalar as dependências e preparar o ambiente, execute:

```bash
uv sync
```

### Executando os Notebooks

Para rodar os experimentos contidos nos notebooks Jupyter:

```bash
uv run jupyter lab
```

Ou, se preferir o Jupyter Notebook clássico:

```bash
uv run jupyter notebook
```

### Executando Scripts

Você também pode executar scripts diretamente através do ambiente gerenciado pelo `uv`:

```bash
uv run python classificao_padroes/modeling/train.py
```

## Relatórios

Os relatórios técnicos de cada trabalho estão localizados na pasta `docs/` em formato LaTeX. Para compilá-los, utilize seu editor LaTeX de preferência ou ferramentas de linha de comando como o `pdflatex`.

## Referências

Os enunciados originais dos trabalhos podem ser encontrados em `references/`:
- `TC1_RecPad-PPGETI_2025.2.pdf`
- `TC2_RecPad-PPGETI_2025.1.pdf`
- `TC3_RecPad-PPGETI_2025.pdf`
