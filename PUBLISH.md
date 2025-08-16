# Build & Publish Guide

Based on the [official uv guide](https://docs.astral.sh/uv/guides/package/)

Hereafter it is assumed that all required code is in `./quatica` folder, and all the necessary data is exported in `./quatica/__init__.py`

## Building

### 1. Change version

The current version can be found in `pyproject.toml`

TRo change the version:

```bash
uv version <new version>
# current version => new version

# Example: uv version 1.0.0
```

### 2. Build package

Firstly, clear the folder `dist` if exists:

```bash
rm -rf ./dist
```

And then build:

```bash
uv build
```

After executing the command above, the `dist` directory with built module will be created

### 3. [OPTIONAL] Local test

To locally instal module:

```bash
pip install dist/quatica-<version>-py3-none-any.whl
```

And the test it:

```bash
python -c "import quatica; print(quatica.__all__)"
```

This should show all the exported modulus. It everything OK, the package can be uninstalled locally:

```bash
pip uninstall dist/quatica-<version>-py3-none-any.whl -y
```

## Publishing

### 1. Authentication

You will need a API Token to publish a library. You can create one in [PyPI account page](https://pypi.org/manage/account/).

### 2. Publish package

```bash
uv publish --token <PyPI token>
```

### 3. [OPTIONAL] Import test

To instal module:

```bash
pip install quatica
```

And the test it:

```bash
python -c "import quatica; print(quatica.__all__)"
```

This should show all the exported modulus. It everything OK, the package can be uninstalled locally:

```bash
pip uninstall quatica -y
```
