# Documentation #
TorchSat Documentation is supported by [Sphinx](http://www.sphinx-doc.org/en/stable/). 
To build the docs, run from the toplevel directory:
```
make docs
```

## Installation ##
```
pip install -r requirements.txt
```

## Workflow ##
To change the documentation, update the `*.rst` files in `source`.

To build the docstrings, `sphinx-apidoc [options] -o <output_path> <module_path> [exclude_pattern, ...]`.
> For torchsat, it should be `sphinx-apidoc --tocfile api -H 'API Reference' -o source/api ../torchsat`.

To build the html pages, `make html`