"""重排序逻辑测试：mock 掉 HTTP 请求，验证按相关度降序返回原下标。"""

from src import config, rerank


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_post(results):
    def post(url, headers, json, timeout):
        assert "rerank" in url
        return _FakeResp({"results": results})

    return post


def test_rerank_orders_by_score(monkeypatch):
    monkeypatch.setattr("src.rerank.requests.post", _fake_post([
        {"index": 0, "relevance_score": 0.2},
        {"index": 1, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.5},
    ]))
    assert rerank.rerank("q", ["a", "b", "c"], top_n=3) == [1, 2, 0]


def test_rerank_empty_documents():
    assert rerank.rerank("q", []) == []


def test_rerank_top_n_limits(monkeypatch):
    monkeypatch.setattr("src.rerank.requests.post", _fake_post([
        {"index": 0, "relevance_score": 1.0},
        {"index": 2, "relevance_score": 0.5},
    ]))
    assert rerank.rerank("q", ["a", "b", "c"], top_n=2) == [0, 2]


def test_rerank_model_from_config():
    assert config.RERANK_MODEL == "BAAI/bge-reranker-v2-m3"
