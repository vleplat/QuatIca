# Build & Publish Guide

Based on the [official uv guide](https://docs.astral.sh/uv/guides/package/)

Hereafter it is assumed that all required code is in `./quatica` folder, and all the necessary data is exported in `./quatica/__init__.py`

## Changing version

The current version can be found in `pyproject.toml`

To change the version:

```bash
uv version <new version>
# current version => new version

# Example: uv version 1.0.0
```

Or using `--bump`:

```bash
uv version --bump patch
# 1.3.0 => 1.3.1
uv version --bump minor
# 1.3.0 => 1.4.0
uv version --bump major
# 1.3.0 => 2.3.0
```

Also you can add `alpha` or `beta` mark:

```bash
uv version --bump alpha
# 1.3.0 => 1.3.1a1
uv version --bump minor --bump beta
# 1.3.0 => 1.4.0b1
```

## GitHub Actions Approach (Recommended)

### 1. Push new tag

Create and push push some release tag to the `main` branch:

```bash
git tag <new version> && git push --tags
# e.g. git tag v1.0.0 && git push --tags
```

**Note**: Version tag should start with **v** symbol

## Manual Approach

### Building

#### 1. Build package

Firstly, clear the folder `dist` if exists:

```bash
rm -rf ./dist
```

And then build:

```bash
uv build
```

After executing the command above, the `dist` directory with built module will be created

#### 2. [OPTIONAL] Local test

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

### Publishing

#### 1. Authentication

You will need a API Token to publish a library. You can create one in [PyPI account page](https://pypi.org/manage/account/).

#### 2. Publish package

```bash
uv publish --token <PyPI token>
```

#### 3. [OPTIONAL] Import test

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
