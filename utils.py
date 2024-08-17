from typing import Dict


def combine_queries(messages: Dict[any, any]):
        system_queries = []
        user_queries = []
        for entry in messages:
            if entry.get("role", "") == "system":
                system_queries.append(entry['content'])
            if entry.get("role", "") == "user":
                user_queries.append(entry['content'])
        return system_queries, user_queries
