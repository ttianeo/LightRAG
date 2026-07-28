from __future__ import annotations

import json

import pytest

from lightrag.operate import (
    _parse_entity_merge_mapping,
    _rebuild_single_entity,
    _rebuild_single_relationship,
    merge_extracted_entities_with_llm,
)
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.prompt import PROMPTS


class _CharTokenizer:
    def encode(self, text: str) -> list[str]:
        return list(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


class _RebuildGraph:
    def __init__(self):
        self.nodes = {
            "International Business Machines": {
                "entity_id": "International Business Machines",
                "entity_type": "organization",
                "description": "stale entity description",
                "source_id": "c1",
                "file_path": "doc.txt",
                "aliases": "IBM",
            },
            "Partner": {
                "entity_id": "Partner",
                "entity_type": "organization",
                "description": "Partner",
                "source_id": "c1",
                "file_path": "doc.txt",
                "aliases": "",
            },
        }
        self.edges = {
            ("International Business Machines", "Partner"): {
                "description": "stale relation description",
                "keywords": "stale",
                "weight": 1.0,
                "source_id": "c1",
                "file_path": "doc.txt",
            }
        }

    async def get_node(self, name):
        return self.nodes.get(name)

    async def has_node(self, name):
        return name in self.nodes

    async def upsert_node(self, name, node_data):
        self.nodes[name] = dict(node_data)

    async def get_edge(self, source, target):
        return self.edges.get((source, target)) or self.edges.get((target, source))

    async def upsert_edge(self, source, target, edge_data):
        self.edges[(source, target)] = dict(edge_data)


class _RebuildStore:
    def __init__(self):
        self.data = {}

    async def upsert(self, data):
        self.data.update(data)


def _node(name: str, description: str) -> dict:
    return {
        "entity_name": name,
        "entity_type": "organization",
        "description": description,
        "source_id": f"chunk-{name}",
        "file_path": "doc.txt",
        "timestamp": 1,
    }


def _edge(source: str, target: str, description: str) -> dict:
    return {
        "src_id": source,
        "tgt_id": target,
        "weight": 1.0,
        "description": description,
        "keywords": "related",
        "source_id": f"chunk-{source}",
        "file_path": "doc.txt",
        "timestamp": 1,
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_llm_entity_merge_uses_one_bounded_call_and_rewrites_graph_inputs():
    calls = []

    async def fake_llm(prompt: str, system_prompt=None, **kwargs):
        calls.append((prompt, system_prompt, kwargs))
        return json.dumps(
            {
                "groups": [
                    {
                        "canonical": "International Business Machines",
                        "members": [
                            "IBM",
                            "International Business Machines",
                        ],
                    }
                ]
            }
        )

    nodes = {
        "IBM": [_node("IBM", "IBM is a technology company.")],
        "International Business Machines": [
            _node(
                "International Business Machines",
                "International Business Machines is also known as IBM.",
            )
        ],
        "Partner": [_node("Partner", "Partner is another company.")],
    }
    edges = {
        ("IBM", "Partner"): [_edge("IBM", "Partner", "IBM works with Partner.")],
        ("International Business Machines", "Partner"): [
            _edge(
                "International Business Machines",
                "Partner",
                "International Business Machines works with Partner.",
            )
        ],
        ("IBM", "International Business Machines"): [
            _edge(
                "IBM",
                "International Business Machines",
                "IBM is an abbreviation.",
            )
        ],
    }
    config = {
        "enable_llm_entity_merge": True,
        "entity_merge_max_entities": 100,
        "entity_merge_description_tokens": 24,
        "tokenizer": _CharTokenizer(),
        "role_llm_funcs": {"extract": fake_llm},
        "llm_cache_identities": {},
    }

    resolved_nodes, resolved_edges, mapping = (
        await merge_extracted_entities_with_llm(nodes, edges, config)
    )

    assert len(calls) == 1
    prompt, system_prompt, kwargs = calls[0]
    payload = json.loads(prompt[prompt.index("{") :])
    assert [entity["name"] for entity in payload["entities"]] == [
        "IBM",
        "International Business Machines",
        "Partner",
    ]
    assert all(len(entity["description"]) <= 24 for entity in payload["entities"])
    assert "When evidence is ambiguous, keep the entities separate." in system_prompt
    assert kwargs["response_format"] == {"type": "json_object"}

    canonical = "International Business Machines"
    assert mapping == {"IBM": canonical, canonical: canonical}
    assert set(resolved_nodes) == {canonical, "Partner"}
    assert len(resolved_nodes[canonical]) == 2
    assert all(
        node["entity_name"] == canonical for node in resolved_nodes[canonical]
    )
    assert all(node["aliases"] == "IBM" for node in resolved_nodes[canonical])
    assert set(resolved_edges) == {(canonical, "Partner")}
    assert len(resolved_edges[(canonical, "Partner")]) == 2
    assert all(
        edge["src_id"] == canonical
        for edge in resolved_edges[(canonical, "Partner")]
    )


def test_entity_merge_response_rejects_invented_and_overlapping_groups():
    result = json.dumps(
        {
            "groups": [
                {"canonical": "A", "members": ["A", "B"]},
                {"canonical": "B", "members": ["B", "C"]},
                {"canonical": "A", "members": ["A", "INVENTED"]},
                {"canonical": "C", "members": ["C"]},
            ]
        }
    )

    assert _parse_entity_merge_mapping(result, {"A", "B", "C"}) == {}


def test_entity_merge_prompt_is_english_and_requires_exact_input_names():
    prompt = PROMPTS["entity_merge_system_prompt"]
    assert "canonical name must be copied verbatim from the input" in prompt
    assert "Do not merge entities merely because they are related" in prompt
    assert not any("\u4e00" <= character <= "\u9fff" for character in prompt)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cached_rebuild_matches_raw_entity_and_relation_aliases():
    graph = _RebuildGraph()
    entity_vdb = _RebuildStore()
    relationship_vdb = _RebuildStore()
    entity_chunks = _RebuildStore()
    relation_chunks = _RebuildStore()
    config = {
        "tokenizer": _CharTokenizer(),
        "source_ids_limit_method": "KEEP",
        "max_source_ids_per_entity": 10,
        "max_source_ids_per_relation": 10,
        "max_file_paths": 10,
        "force_llm_summary_on_merge": 8,
        "summary_context_size": 10_000,
        "summary_max_tokens": 10_000,
    }
    canonical = "International Business Machines"
    raw_entity = _node("IBM", "IBM is the long-standing technology company.")
    raw_relation = _edge("IBM", "Partner", "IBM works with Partner.")

    await _rebuild_single_entity(
        graph,
        entity_vdb,
        canonical,
        ["c1"],
        {"c1": {"IBM": [raw_entity]}},
        None,
        config,
        entity_chunks_storage=entity_chunks,
    )
    await _rebuild_single_relationship(
        graph,
        relationship_vdb,
        entity_vdb,
        canonical,
        "Partner",
        ["c1"],
        {"c1": {("IBM", "Partner"): [raw_relation]}},
        None,
        config,
        relation_chunks_storage=relation_chunks,
        entity_chunks_storage=entity_chunks,
    )

    assert graph.nodes[canonical]["description"] == raw_entity["description"]
    entity_record = next(iter(entity_vdb.data.values()))
    assert f"Aliases: IBM{GRAPH_FIELD_SEP}" not in entity_record["content"]
    assert "Aliases: IBM" in entity_record["content"]
    relation = graph.edges[(canonical, "Partner")]
    assert relation["description"] == raw_relation["description"]
