from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.graph import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_graph_stats_api_returns_stable_shape():
    client = _client()
    stats = {
        "node_count": 3,
        "relationship_count": 2,
        "node_labels": [{"label": "Entity", "count": 2}],
        "relationship_types": [{"type": "RELATES_TO", "count": 2}],
    }

    with patch("api.routers.graph.GraphStore") as MockStore:
        inst = MagicMock()
        inst.get_graph_stats.return_value = stats
        MockStore.return_value = inst

        resp = client.get("/graph/stats")

    assert resp.status_code == 200
    data = resp.json()
    assert data["node_count"] == 3
    assert data["relationship_types"][0]["type"] == "RELATES_TO"
    inst.close.assert_called_once()


def test_graph_subgraph_api_passes_filters_and_returns_nodes_edges():
    client = _client()
    subgraph = {
        "nodes": [
            {
                "id": "n1",
                "label": "BERT",
                "type": "Entity",
                "labels": ["Entity"],
                "properties": {"name": "BERT"},
            }
        ],
        "edges": [
            {
                "id": "r1",
                "source": "n1",
                "target": "n1",
                "type": "RELATES_TO",
                "label": "RELATES_TO",
                "properties": {},
            }
        ],
    }

    with patch("api.routers.graph.GraphStore") as MockStore:
        inst = MagicMock()
        inst.get_subgraph.return_value = subgraph
        MockStore.return_value = inst

        resp = client.get("/graph/subgraph?limit=50&node_types=Entity&search=BERT")

    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"][0]["label"] == "BERT"
    assert data["edges"][0]["type"] == "RELATES_TO"
    inst.get_subgraph.assert_called_once_with(
        limit=50,
        node_types=["Entity"],
        search="BERT",
    )
    inst.close.assert_called_once()
