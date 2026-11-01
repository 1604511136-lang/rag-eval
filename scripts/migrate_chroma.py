"""把已有的 embeddings_{size}.npy 迁移到 Chroma 向量库（无需重新调用 embedding API）。

用法：python scripts/migrate_chroma.py [--corpus fastapi-zh]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config, ingest, vector_store


def main() -> None:
    ap = argparse.ArgumentParser(description="把 .npy 向量迁移到 Chroma")
    ap.add_argument("--corpus", default=config.CORPUS_NAME)
    args = ap.parse_args()
    config.use_corpus(args.corpus)

    for size in config.CHUNK_SIZES:
        emb_path = config.CHUNK_DIR / f"embeddings_{size}.npy"
        if not emb_path.exists():
            print(f"[skip] chunk_size={size}: 未找到 {emb_path}")
            continue
        chunks = ingest.load_chunks(size)
        emb = np.load(emb_path).astype("float32").tolist()
        vector_store.upsert(size, chunks, emb)
        print(f"[ok] chunk_size={size}: 已写入 {len(chunks)} 个向量到 "
              f"{vector_store.collection_name(size)}")


if __name__ == "__main__":
    main()
