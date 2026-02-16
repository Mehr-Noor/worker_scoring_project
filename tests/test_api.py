import pytest

def test_api_import():
    from src.api import app
    assert app is not None
