import pytest
from neurostore import utils
import os


def test_safe_get():
    example_dict = {"first": 1}
    res1 = utils.safe_get(example_dict, 0, "first")
    res2 = utils.safe_get(example_dict, 0, "sec")
    assert res1 == 1
    assert res2 == 0


def test_safe_get_complex():
    example_dict = {"first": {"nested_first": 1}, "sec": 2}
    res1 = utils.safe_get(example_dict, 0, "first", "nested_first")
    res2 = utils.safe_get(example_dict, 0, "sec")
    res3 = utils.safe_get(example_dict, 0, "first", "nested_sec")
    assert res1 == 1
    assert res2 == 2
    assert res3 == 0


def test_safe_config_import(change_dir, monkeypatch):
    os.chdir("data")
    expected = {
        "database_path": "neurostore_config/neurostore.db",
        "api_keys": {"OPENAI": "your_api_key_here"},
        "embedding_models": {"query": "Default", "answer": "text-embedding-3-small"},
    }
    actual = utils.config_import()
    assert expected == actual
