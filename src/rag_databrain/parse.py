from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

# 字符窗口，不是 token；重叠避免句子被切两半后两边都检索不到。
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 80


def chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    # 先压空白，避免换行把同一句拆成大量空 chunk。
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks


def read_file(name: str, raw: bytes) -> str:
    # 只认三种后缀，别的格式交给上层 400，不要在这里猜。
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        try:
            pages = PdfReader(BytesIO(raw)).pages
            text = "\n".join((p.extract_text() or "") for p in pages)
        except Exception as e:
            raise ValueError("无法解析 PDF") from e
    elif suffix in {".txt", ".md"}:
        text = _decode_text(raw)
    else:
        raise ValueError("只支持 .txt .md .pdf")

    if not text.strip():
        raise ValueError("没有可提取的文本")
    return text


def _decode_text(raw: bytes) -> str:
    # Windows 中文文件经常是 gbk，utf-8 失败再试一次。
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("文本编码无法识别")
