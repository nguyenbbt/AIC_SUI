from fastapi.testclient import TestClient

from retrieval_api.demo import app


def test_demo_supports_complete_ui_smoke_flow() -> None:
    client = TestClient(app)
    assert client.get("/health/ready").json()["checks"]["demo"] == "true"
    assert client.get("/catalog/object-labels").json()["source"] == "demo_fixture"
    search = client.post("/search", json={"query": "person", "query_id": "q1"}).json()
    assert len(search["candidates"]) == 12
    frame_id = search["candidates"][0]["frame_id"]
    assert client.get(f"/media/keyframes/{frame_id}").headers["content-type"].startswith("image/svg+xml")
    assert len(client.get(f"/media/keyframes/{frame_id}/neighbors").json()["frames"]) == 3
    rewrite = client.post(
        "/query/rewrite",
        json={"query": "Một người mặc áo đỏ đứng cạnh ô tô", "request_id": "r1"},
    ).json()
    assert rewrite["paraphrases"] == [
        "Một người mặc một chiếc áo màu đỏ đang đứng bên cạnh một chiếc xe ô tô",
        "a person wearing a red shirt standing next to a car",
    ]
    assert client.post("/trake", json={"query_id": "t1", "event_texts": ["a", "b"]}).json()["results"]
    assert client.post("/vqa", json={"question_id": "v1", "question": "ai?", "answer_type": "short_text"}).json()["result"]["response"]["status"] == "answered"
