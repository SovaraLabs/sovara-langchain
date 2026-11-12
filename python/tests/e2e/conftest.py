"""Run these tests against a real Sovara exec server: ``uv run pytest -m e2e``.

Plain ``pytest`` leaves them out; they are skipped without a sovara checkout.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.support import Backend, run_backend, sovara_source


def pytest_collection_modifyitems(items):
    here = Path(__file__).parent
    for item in items:
        if here in Path(item.fspath).parents:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def backend() -> Iterator[Backend]:
    source = sovara_source()
    if source is None:
        pytest.skip("End-to-end tests need a sovara checkout; set SOVARA_SOURCE_DIR.")
    yield from run_backend(source)


@pytest.fixture
def server(backend):
    """Yield the server and a client for a fresh project."""
    backend.sent_steps.clear()
    client = backend.client()
    try:
        yield backend, client
    finally:
        client.close()
