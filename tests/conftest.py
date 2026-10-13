import pytest

from prv.data import load_toy_data


@pytest.fixture(scope="session")
def toy():
    return load_toy_data()
