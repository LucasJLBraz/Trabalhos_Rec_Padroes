# Trabalhos_Rec_Padroes

Master's project repository for Pattern Recognition implementations.

## Project Organization

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   └── trabalhos_rec_padroes
│       ├── __init__.py    <- Makes src a Python module
│       │
│       ├── data           <- Scripts to download or generate data
│       │   └── make_dataset.py
│       │
│       ├── features       <- Scripts to turn raw data into features for modeling
│       │   └── build_features.py
│       │
│       ├── models         <- Scripts to train models and then use trained models to make
│       │   │                 predictions
│       │   ├── predict_model.py
│       │   └── train_model.py
│       │
│       └── visualization  <- Scripts to create exploratory and results oriented visualizations
│           └── visualize.py
│
└── tests              <- Test files
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository
```bash
git clone https://github.com/LucasJLBraz/Trabalhos_Rec_Padroes.git
cd Trabalhos_Rec_Padroes
```

2. Create a virtual environment
```bash
make create_environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
make requirements
```

4. Install the project in development mode
```bash
pip install -e .
```

## Usage

### Running Notebooks

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory to start exploring.

### Project Commands

- `make requirements` - Install Python dependencies
- `make clean` - Delete all compiled Python files
- `make lint` - Lint using flake8
- `make create_environment` - Set up Python virtual environment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>