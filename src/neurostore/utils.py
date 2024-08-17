from typing import Dict, Literal, Union, Any, List
import os
import json


def safe_get(info: Dict[str, Any], default: Any = None, *args: Any) -> Any:
    for arg in args:
        if isinstance(info, dict) and arg in info:
            info = info[arg]
        else:
            return default
    return info


def config_import() -> Dict[str, str]:
    if not os.path.exists("neurostore_config.json"):
        return {}
    with open("neurostore_config.json", "r") as f:
        info = json.load(f)
    return info


def combine_queries(messages: List[Dict[str, str]]):
    system_queries = []
    user_queries = []
    for entry in messages:
        if entry.get("role", "") == "system":
            system_queries.append(entry["content"])
        if entry.get("role", "") == "user":
            user_queries.append(entry["content"])
    return system_queries, user_queries
