# Build & Publish Guide

Based on the [official uv guide](https://docs.astral.sh/uv/guides/package/)

Hereafter it is assumed that all required code is in `./quatica` folder, and all the necessary data is exported in `./quatica/__init__.py`

There are **three main approaches** of publishing:

- [Run the corresponding script](#publishing-via-script-recommended)
- [Change version](#changing-version) + [GitHub Actions](#github-actions-approach)
- [Change version](#changing-version) + [Manual approach](#manual-approach)

## Publishing via Script (Recommended)

There is special script `publish.py` to automate publishing process.

### Examples of usage

**Example: Perform a dry run to see what would happen for a patch release:**

```bash
uv run ./publish.py --bump patch --dry-run
```

_Expected Output:_

```text
INFO: Checking prerequisites...
SUCCESS: Prerequisites met.
EXEC: uv version --bump patch
INFO: Dry run mode: Assuming new version is '1.2.3'.
EXEC: git tag v1.2.3
EXEC: git push origin v1.2.3
SUCCESS: Successfully tagged version v1.2.3.
SUCCESS: Tag pushed to remote. The GitHub Action should now trigger the publish process.
```

**Example: Bump the `minor` version and add a `beta` tag:**

```bash
./publish.py --bump minor --prerelease beta
```

_This will:_

1. Run `uv version --bump minor --bump beta`.
2. Find the new version (e.g., `1.4.0b1`).
3. Run `git add pyproject.toml uv.lock`.
4. Run `git commit -m "Bump version to 1.4.0b1`.
5. Run `git tag v1.4.0b1`.
6. Run `git push`.
7. Run `git push origin v1.4.0b1`.

**Example: Set a specific version and create the tag, but don't push it yet:**

```bash
./publish.py --set-version 2.0.0 --no-push
```

_This will:_

1. Run `uv version 2.0.0`.
2. Run `git tag v2.0.0`.
3. Stop without pushing, reminding you to do it manually.

This script provides a robust and user-friendly way to manage your package releases directly from the command line, perfectly aligning with the workflow described in your `README`.

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

## GitHub Actions Approach

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
