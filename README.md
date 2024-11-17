# Crypto portfolios

This code repository accompanies the paper [Simple and Effective Portfolio Construction with Crypto Assets](https://web.stanford.edu/~boyd/papers/crypto_portfolio.html).


## Poetry

We assume you share the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
make install
```

to replicate the virtual environment we have defined in [pyproject.toml](pyproject.toml)
and locked in [poetry.lock](poetry.lock).

## Jupyter

We install [JupyterLab](https://jupyter.org) on fly within the aforementioned
virtual environment. Executing

```bash
make jupyter
```

will install and start the jupyter lab.
