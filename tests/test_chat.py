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


def test_create_db_empty_dir(tmp_path):
    os.chdir("empty_dir")
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
def test_create_db_config_file(tmp_path):
    os.chdir("data")
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


@pytest.fixture
def create_neurostore(tmp_path):
    model = chat.Neurostore(
        db_path=os.path.join(tmp_path),
        embedding_model_query="Default",
        embedding_model_answer="Default",
        api_key="random_key",
    )
    return model


def test_inference(create_neurostore):

    # make sure system queries are there
    assert len(create_neurostore.db.list_collections()) == 1

    message_quantum = [{"role": "user", "content": "What is quantum entanglement?"}]
    message_penguins1 = [{"role": "user", "content": "Where can I find penguins?"}]
    message_penguins2 = [
        {"role": "user", "content": "Tell me where penguins are located?"}
    ]
    create_neurostore.create(
        messages=message_quantum, store=True, model="gpt-3.5-turbo-1106", max_tokens=10
    )
    create_neurostore.create(
        messages=message_penguins1,
        store=True,
        model="gpt-3.5-turbo-1106",
        max_tokens=10,
    )
    create_neurostore.create(
        messages=message_penguins2,
        store=True,
        model="gpt-3.5-turbo-1106",
        max_tokens=10,
    )
    # should be different enough queries?
    assert len(create_neurostore.db.list_collections()) == 3

    res = create_neurostore.query(messages=message_penguins2, num_results=2)[0]

    assert len(res) == 2

    for item in res:
        assert ("penguin" in item["entity"]["response"]) or (
            "Penguin" in item["entity"]["response"]
        )
