# Getting started
Clone the repo in your working machine (usually Google Cloud instance).


```bash
make env
source env/bin/activate
make init
```
From there, you should have a folder named **io/** at the root of the project. This folder is a symlink to **/io/** so you can share the raw (and processd) data as well as the resulting models.


If the Google bucket has not been pulled already, or data in bucket has been updated recently:

```bash
make sync_raw_data
```


# Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── io                 <- All input/output files, it is a symlink to the attached disk /io
    |   ├── data
    |   │   ├── external   <- Data from third party sources.
    |   │   ├── interim    <- Intermediate data that has been transformed.
    |   │   ├── processed  <- The final, canonical data sets for modeling.
    |   │   └── raw        <- The original, immutable data dump.
    |   ├── models         <- Trained and serialized models, model predictions, or model summaries
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── api            <- Scripts to serve model predictions via API
    │   │   └── server.py
    │   |
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── tests         <- Tests for src content
    │   │   ├── test_all.py
    │   │   └── test_api.py
    │   │   └── test_data.py
    │   │   └── test_models.py
    │   │   └── test_visualization.py
    |   |
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

