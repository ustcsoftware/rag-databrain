import logging
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

# 字符窗口，不是 token；重叠避免句子被切两半后两边都检索不到。
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 80
# 平均每页少于此值就当图片或扫描件。ponytail: 文本层是长乱码时不会识别；正文里夹的图也不识别。双栏和表格再换版面解析。
_SCAN_CHARS_PER_PAGE = 40
_ocr_engine = None


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
        text = _read_pdf(raw)
    elif suffix in {".txt", ".md"}:
        text = _decode_text(raw)
    else:
        raise ValueError("只支持 .txt .md .pdf")

    if not text.strip():
        raise ValueError("没有可提取的文本")
    return text


def _read_pdf(raw: bytes) -> str:
    # 先信文本层。整份平均每页不够字，再渲染识别。
    try:
        pages = PdfReader(BytesIO(raw)).pages
        parts = [(p.extract_text() or "").strip() for p in pages]
    except Exception as e:
        raise ValueError("无法解析 PDF") from e

    chars = sum(len("".join(t.split())) for t in parts)
    if parts and chars < _SCAN_CHARS_PER_PAGE * len(parts):
        scanned = _ocr_pdf(raw)
        scanned_chars = sum(len("".join(t.split())) for t in scanned)
        # 识别结果更短就留文本层，避免短正文被空识别结果盖掉。
        if scanned_chars > chars:
            parts = scanned

    return "\n".join(t for t in parts if t)


def _ocr_pdf(raw: bytes) -> list[str]:
    # 模型只在真的碰到扫描件时加载。scale=2 约 144 DPI，小字扫不清再调高。
    global _ocr_engine
    try:
        import pypdfium2 as pdfium
        from rapidocr import RapidOCR
    except ImportError as e:
        raise ValueError("无法识别扫描页") from e

    if _ocr_engine is None:
        try:
            _ocr_engine = RapidOCR()
        except Exception as e:
            raise ValueError("无法识别扫描页") from e
        # 库自己打模型路径，入库日志会被刷屏。
        logging.getLogger("rapidocr").setLevel(logging.ERROR)

    parts = []
    try:
        doc = pdfium.PdfDocument(raw)
    except Exception as e:
        raise ValueError("无法识别扫描页") from e

    try:
        for i in range(len(doc)):
            page = doc[i]
            bitmap = page.render(scale=2)
            try:
                result = _ocr_engine(bitmap.to_pil(), text_score=0.5)
            finally:
                bitmap.close()
                page.close()
            if result.txts:
                parts.append(" ".join(result.txts))
    except Exception as e:
        raise ValueError("无法识别扫描页") from e
    finally:
        doc.close()
    return parts


def _decode_text(raw: bytes) -> str:
    # Windows 中文文件经常是 gbk，utf-8 失败再试一次。
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("文本编码无法识别")
