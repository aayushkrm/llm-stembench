"""Shared fixtures: budget-file isolation and headless matplotlib.

Every test runs with STEMBENCH_BUDGET_DIR (and the module-level BUDGET_DIR constant
inside stembench.providers.openai_compat, which is resolved at tracker-construction
time) pointed at a per-test tmp dir, so the suite never touches data/cache.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_budget_dir(tmp_path, monkeypatch):
    budget_dir = tmp_path / "budget-cache"
    monkeypatch.setenv("STEMBENCH_BUDGET_DIR", str(budget_dir))
    import stembench.providers.openai_compat as oai

    monkeypatch.setattr(oai, "BUDGET_DIR", budget_dir)
    yield budget_dir


@pytest.fixture(autouse=True)
def headless_matplotlib(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    yield
