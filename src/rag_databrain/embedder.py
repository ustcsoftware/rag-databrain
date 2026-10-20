import os

from openai import OpenAI

# 上游一次请求条数上限远大于此；64 只是避免单次 body 过大。
_BATCH = 64


def openai_client() -> OpenAI:
    # 兼容中转只改 base_url，不要为每家 SDK 再包一层。
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("缺少 OPENAI_API_KEY")

    kwargs = {"api_key": key}
    base = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base:
        kwargs["base_url"] = base
    return OpenAI(**kwargs)


def embed_texts(texts: list[str]) -> list[list[float]]:
    # 存储层和检索只收向量，模型名和协议留在这一层。
    if not texts:
        return []

    client = openai_client()
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    out = []
    for i in range(0, len(texts), _BATCH):
        resp = client.embeddings.create(model=model, input=texts[i : i + _BATCH])
        out.extend(d.embedding for d in resp.data)
    return out
