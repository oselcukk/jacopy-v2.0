"""Smoke-execute every Jupyter notebook under ``docs/tutorials/``.

Each notebook is run end-to-end via :class:`nbclient.NotebookClient`;
any cell raising an exception fails the test. This keeps the
tutorials in ``docs/tutorials/`` from bit-rotting silently as the
library API evolves.

Skips entirely when ``nbformat`` / ``nbclient`` aren't installed,
they're optional extras under ``[project.optional-dependencies.docs]``
so contributors who don't touch docs don't need to install them.

The test harness prepends the repo root to ``PYTHONPATH`` for the
spawned kernel so the notebooks can ``import jacopy`` without
requiring the package to be installed into the kernel's site-packages.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient_mod = pytest.importorskip("nbclient")

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = REPO_ROOT / "docs" / "tutorials"


def _discover_notebooks() -> list[Path]:
    if not TUTORIAL_DIR.is_dir():
        return []
    return sorted(TUTORIAL_DIR.glob("*.ipynb"))


NOTEBOOKS = _discover_notebooks()


@pytest.fixture
def kernel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the spawned kernel can ``import jacopy``.

    The system ``python3`` kernel has no knowledge of this repo's
    checkout; prepending the repo root to ``PYTHONPATH`` is the
    portable fix that doesn't require a site-wide install.
    """
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT)]
    if existing:
        parts.append(existing)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(parts))


@pytest.mark.parametrize(
    "notebook_path",
    NOTEBOOKS,
    ids=[nb.stem for nb in NOTEBOOKS] or ["<no_notebooks>"],
)
def test_notebook_executes(notebook_path: Path, kernel_env: None) -> None:
    """Run every cell; any raised exception fails the test."""
    if not NOTEBOOKS:
        pytest.skip("no tutorial notebooks found")
    nb = nbformat.read(notebook_path, as_version=4)
    client = nbclient_mod.NotebookClient(
        nb,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
