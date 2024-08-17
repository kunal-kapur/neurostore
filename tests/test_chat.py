import pytest
from neurostore import chat, utils
import os
from neurostore.embedding_models import (
    EmbeddingCompletionMapping,
    EmbeddingModel,
    AlbertSmallV2,
    OpenAI_Small,
    OpenAI_Large,
)
import json
from unittest import mock


@pytest.mark.parametrize("change_dir", ["empty_dir"], indirect=True)
def test_create_db_empty_dir(tmp_path, change_dir):
    chat.Neurostore(
        db_path=os.path.join(tmp_path, "neurostore_no_config", "neurostore.db"),
        embedding_model_query="Default",
        embedding_model_answer="text-embedding-3-small",
        api_key="random_key",
    )
    expected_results = {
        "embedding model query": "paraphrase-albert-small-v2",
        "embedding model query dimension": 768,
        "embedding model answer": "text-embedding-3-small",
        "embedding model answer dimension": 1536,
    }
    with open(
        os.path.join(tmp_path, "neurostore_no_config", "neurostore_info.json")
    ) as f:
        actual_result = json.load(f)
        for key in expected_results:
            assert actual_result.get(key, "") == expected_results[key]
    assert os.path.exists(
        os.path.join(tmp_path, "neurostore_no_config", "neurostore.db")
    )


@pytest.mark.parametrize("change_dir", ["data"], indirect=True)
def test_create_db_config_file(tmp_path, change_dir):

    modified_config = utils.config_import()
    modified_config["database_path"] = os.path.join(
        tmp_path, modified_config["database_path"]
    )

    with mock.patch("neurostore.utils.config_import") as mocked_utils_config_import:
        mocked_utils_config_import.return_value = modified_config
        chat.Neurostore(
            db_path=os.path.join(tmp_path, "neurostore_fake", "neurostore_fake.db"),
            embedding_model_query="text-embedding-3-large",
            embedding_model_answer="text-embedding-3-large",
        )

    expected_results = {
        "embedding model query": "paraphrase-albert-small-v2",
        "embedding model query dimension": 768,
        "embedding model answer": "text-embedding-3-small",
        "embedding model answer dimension": 1536,
    }
    with open(os.path.join(tmp_path, "neurostore_config", "neurostore_info.json")) as f:
        actual_result = json.load(f)
    for key in expected_results:
        assert actual_result.get(key, "") == expected_results[key]
    assert os.path.exists(os.path.join(tmp_path, "neurostore_config", "neurostore.db"))
