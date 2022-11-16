import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )
    parser.addoption(
        "--run-dist",
        action="store_true",
        default=False,
        help="run distributed tests, requires 2 devices minimum!",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test to run"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as too slow for github actions"
    )
    config.addinivalue_line(
        "markers",
        (
            "dist: mark test as a distributed test requiring at least 2"
            "devices"
        ),
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(
            reason="need --run-integration option to run"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--run-dist"):
        skip_dist = pytest.mark.skip(reason="need --run-dist option to run")
        for item in items:
            if "dist" in item.keywords:
                item.add_marker(skip_dist)
