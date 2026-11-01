from src.ingest import _source_to_url, _split_long, _split_sections, chunk_markdown


MD = """# 入门

这是第一段正文。

## 安装

安装 fastapi。

```python
# 这是代码注释，不是标题
print("hi")
```

### 详细步骤

第一步。
"""


def test_split_sections_ignores_code_fence():
    sections = _split_sections(MD)
    titles = [t for t, _ in sections]
    assert "入门" in titles[0]
    assert any("入门 > 安装 > 详细步骤" in t for t in titles)
    # 代码块内的 # 不能成为新标题
    assert not any("这是代码注释" in t for t in titles)


def test_split_long_no_split_when_short():
    assert _split_long("hello", 100, 10) == ["hello"]


def test_split_long_overlap():
    pieces = _split_long("abcdefghij", 4, 2)
    # step = 4-2 = 2 -> ["abcd", "cdef", "efgh", "ghij"]
    assert pieces == ["abcd", "cdef", "efgh", "ghij"]


def test_chunk_markdown_metadata():
    chunks = chunk_markdown("# 标题\n\n正文内容", "docs/zh/docs/x.md", "http://x/", 100, 20)
    assert chunks
    assert chunks[0]["section"] == "标题"
    assert chunks[0]["source"] == "docs/zh/docs/x.md"
    assert chunks[0]["url"] == "http://x/"
    assert "正文内容" in chunks[0]["text"]


def test_source_to_url():
    assert _source_to_url("docs/zh/docs/tutorial/first-steps.md") == \
        "https://fastapi.tiangolo.com/zh/tutorial/first-steps/"
    assert _source_to_url("docs/zh/docs/index.md") == "https://fastapi.tiangolo.com/zh/"
