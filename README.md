# Credit Risk Classifier with GitHub Actions

This project presents a credit risk classification pipeline using Python and scikit-learn, with process automation through GitHub Actions.

The repository was developed as a practical example to demonstrate fundamental Continuous Integration and Continuous Delivery (CI/CD) features, including automated test execution, model training and evaluation, and automatic version release.

## Objective

The project's objective is to train a classification model to predict credit card default from tabular customer attributes, using an automated workflow for validation and versioning.

## Features

- tabular data preprocessing;
- machine learning model training and evaluation;
- pipeline integration tests;
- CI workflow with GitHub Actions;
- automatic version release by tag;
- generation of artifacts such as the trained model and metrics reports.

## Project Structure

```text

├── .github/workflows/
│ ├── ci.yml
│ └── release.yml
├── configs/
│ ├── data_cfg.py
│ └── model_cfg.py
├── date/
│ └── raw/
│  └── amex_data.csv
├── models/
│ └── model.joblib
├── reports/
│ ├── full_report.json
│ └── metrics.json
├── src/
│ ├── train.py
│ ├── preprocessing.py
│ └── __init__.py
├── tests/
│ └── test_pipeline.py
├── pyproject.toml
└── README.md
```

## Technologies
- Python
- Pandas
- Numpy
- Scikit-learn
- Pytest
- Joblib
- GitHub Actions

## How to run locally

### Clone the repository:

```bash
git clone https://github.com/leomsfreitas/CreditRiskClassifier
```

### Install dependencies:

```bash
pip install -e ".[dev]"
```

### Run the training:

```bash
python src/train.py
```

## How to run the tests

```bash
pytest tests/ -v
```

## Automation with GitHub Actions

The project uses GitHub Actions to automate essential steps in the development and delivery of the model.

### Continuous Integration (CI)

In the CI workflow, the project automatically executes:

- installation of dependencies;
- execution of automated tests;
- training and evaluation of the model;
- generation of metrics;
- comparison of metrics in pull requests.

### Continuous Delivery (CD)

In the release workflow, the project automatically executes:

- creation of a release from version tags;
- generation of the trained model;
- availability of version reports;
- organization of experiment artifacts.

## Automatic Release

To generate a new version of the project, simply create and send a tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```
This automatically triggers the release workflow.