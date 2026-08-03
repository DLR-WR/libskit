Release Workflow
================

This document describes how to develop, version, and release LIBSkit.

Overview
--------

LIBSkit uses **Conventional Commits** and **Semantic Versioning** to manage releases:

* Developers write commit messages following a standard format
* Version numbers are determined automatically based on commit types
* Releases are triggered manually via GitLab CI/CD
* Wheels are built for Linux, Windows, and macOS automatically

Semantic Versioning
-------------------

Version numbers follow the format ``MAJOR.MINOR.PATCH`` (e.g., ``1.2.3``):

* **MAJOR**: Breaking changes (incompatible API changes)
* **MINOR**: New features (backwards compatible)
* **PATCH**: Bug fixes (backwards compatible)

Conventional Commits
--------------------

Every commit message should follow this format::

    <type>: <description>

    [optional body]

    [optional footer]

Commit Types
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 40 20

   * - Type
     - Description
     - Version Bump
   * - ``fix``
     - Bug fixes
     - Patch (1.0.0 → 1.0.1)
   * - ``feat``
     - New features
     - Minor (1.0.0 → 1.1.0)
   * - ``feat!``
     - Breaking changes
     - Major (1.0.0 → 2.0.0)
   * - ``perf``
     - Performance improvements
     - Patch
   * - ``docs``
     - Documentation only
     - No release
   * - ``refactor``
     - Code refactoring
     - No release
   * - ``test``
     - Adding tests
     - No release
   * - ``chore``
     - Maintenance tasks
     - No release
   * - ``ci``
     - CI/CD changes
     - No release

Examples
~~~~~~~~

**Bug fix** (patch release)::

    fix: correct wavelength calculation in spectrum model

**New feature** (minor release)::

    feat: add support for LTE layers modeling

    This adds a new LTELayers class that allows modeling
    multiple plasma layers with different temperatures.

**Breaking change** (major release)::

    feat!: rename PlasmaModel to LTEModel

    BREAKING CHANGE: The PlasmaModel class has been renamed
    to LTEModel for clarity. Update your imports accordingly.

Setting Up Pre-commit Hooks
---------------------------

We use pre-commit hooks to enforce conventional commit messages locally.
This catches formatting issues before you push, saving CI time.

Installation (one-time setup)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Install pre-commit
    pip install pre-commit

    # Install the git hooks (run from repo root)
    pre-commit install --hook-type commit-msg --hook-type pre-commit

What happens now
~~~~~~~~~~~~~~~~

When you run ``git commit``:

1. **Pre-commit checks** run (trailing whitespace, YAML validation, etc.)
2. **Commit message validation** checks your message format

If your commit message doesn't follow the conventional format, you'll see::

    Conventional commit validation failed.
    commit message: "updated some stuff"

    Expected format: <type>(<scope>): <description>
    Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert

Simply amend your commit with a proper message::

    git commit --amend -m "fix: correct temperature calculation"

Bypassing hooks (emergency only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In rare cases, you can skip hooks with ``--no-verify``::

    git commit --no-verify -m "wip: temporary commit"

.. warning::
   The CI will still validate commits, so non-conventional commits
   will be flagged in merge requests.

Development Workflow
--------------------

1. Create a Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git checkout main
    git pull origin main
    git checkout -b feature/my-new-feature

2. Make Changes with Conventional Commits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Work on your changes...
    git add .
    git commit -m "feat: add new plasma temperature estimation"

    # Fix a bug...
    git add .
    git commit -m "fix: handle edge case in Boltzmann plot"

3. Push and Create Merge Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git push origin feature/my-new-feature

Then create a Merge Request in GitLab.

4. Merge to Main
~~~~~~~~~~~~~~~~

After review, merge the MR into ``main``. The CI pipeline will run tests.

Release Workflow
----------------

Releases are **not automatic** - they require manual triggering for control.

1. Trigger Semantic Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After merging to ``main``:

1. Go to **CI/CD → Pipelines** in GitLab
2. Find the pipeline for your merge commit
3. Click the **Play** button (▶) on the ``semantic-release`` job
4. The job analyzes commits since the last tag and:

   - Determines the version bump (patch/minor/major)
   - Creates a new git tag (e.g., ``v1.2.0``)
   - Updates ``CHANGELOG.md``

2. Build Wheels (Automatic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the tag is created, the pipeline automatically:

1. Builds wheels for all platforms (Linux, Windows, macOS)
2. Builds a source distribution
3. Runs tests on the built wheels
4. Publishes to GitLab Package Registry

3. Verify the Release
~~~~~~~~~~~~~~~~~~~~~

Check the **Packages & Registries → Package Registry** in GitLab to see
the published packages.

Manual Tagging (Alternative)
----------------------------

If you prefer full manual control, you can skip ``semantic-release`` and
create tags directly:

.. code-block:: bash

    # Ensure you're on main with latest changes
    git checkout main
    git pull origin main

    # Create and push a tag
    git tag v1.2.0
    git push origin v1.2.0

This will trigger the wheel building and publishing pipeline.

Installing Released Versions
----------------------------

From GitLab Package Registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install libskit --index-url https://codebase.helmholtz.cloud/api/v4/projects/<PROJECT_ID>/packages/pypi/simple

From PyPI (when published)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install libskit

From Source
~~~~~~~~~~~

.. code-block:: bash

    git clone https://codebase.helmholtz.cloud/christoph.egerland/libskit.git
    cd libskit
    pip install .

Troubleshooting
---------------

No version bump detected
~~~~~~~~~~~~~~~~~~~~~~~~

If ``semantic-release`` says "No release needed", check that your commits
use the conventional format. Commits without ``fix:``, ``feat:``, or ``perf:``
prefixes don't trigger releases.

Pipeline fails on Windows/macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Windows and macOS builds require specific GitLab runners. If unavailable,
the Linux wheels will still be built and published.

Version mismatch
~~~~~~~~~~~~~~~~

The version is determined by git tags via ``setuptools-scm``. If you see
unexpected versions, check:

.. code-block:: bash

    git describe --tags
    git tag --list
