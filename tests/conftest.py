"""Shared fixtures.

Tests that need the real LexGLUE corpus skip, with a reason, when it is not on
disk; everything else runs on a fresh clone with no data at all.
"""

from __future__ import annotations

import pytest

from outcome import corpus


def _split(name: str):
    if not corpus.available():
        pytest.skip(
            f"ECtHR corpus not found in {corpus.data_dir()}; run "
            "`python scripts/fetch_data.py` (or set OUTCOME_DATA) to run this test"
        )
    return corpus.load(name)


@pytest.fixture(scope="session")
def train():
    return _split("train")


@pytest.fixture(scope="session")
def validation():
    return _split("validation")


@pytest.fixture(scope="session")
def test_split():
    return _split("test")


@pytest.fixture(scope="session")
def splits():
    return {s: _split(s) for s in corpus.SPLITS}
