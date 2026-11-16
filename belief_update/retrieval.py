"""Client for the Search-R1 retrieval server (E5 over Wikipedia 2018).

Launch the server from the Search-R1 repo (`retrieval_launch.sh`), then pass
`--retriever_url http://127.0.0.1:8000/retrieve`. Passages are formatted
exactly as in Search-R1 (`Doc i(Title: ...) text`).
"""

from __future__ import annotations

import time

import requests


def passages_to_string(docs: list) -> str:
    out = []
    for i, d in enumerate(docs):
        content = d["document"]["contents"] if "document" in d else d["contents"]
        title, _, text = content.partition("\n")
        out.append(f"Doc {i + 1}(Title: {title}) {text}")
    return "\n".join(out) + "\n"


class SearchR1Retriever:
    def __init__(self, url: str, topk: int = 3, timeout: float = 60.0, retries: int = 3):
        self.url = url
        self.topk = topk
        self.timeout = timeout
        self.retries = retries

    def __call__(self, queries: list[str]) -> list[list[dict]]:
        if not queries:
            return []
        payload = {"queries": queries, "topk": self.topk, "return_scores": True}
        for attempt in range(self.retries):
            try:
                r = requests.post(self.url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["result"]
            except requests.RequestException:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


def doc_record(d: dict) -> dict:
    """Compact form saved in traces: title, text, score."""
    content = d["document"]["contents"] if "document" in d else d["contents"]
    title, _, text = content.partition("\n")
    return {"title": title, "text": text, "score": d.get("score")}
