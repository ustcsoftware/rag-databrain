import logging
import re
from io import BytesIO
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader

from rag_databrain.layout import (
    drop_repeated,
    flatten_headers,
    merge_tables,
    ocr_to_words,
    reading_lines,
    repeated_edges,
    rows_to_markdown,
    table_summary,
    tables_continue,
    tables_from_words,
)

# 字符窗口，不是 token；重叠避免句子被切两半后两边都检索不到。
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 80
# 平均每页少于此值就当图片或扫描件。ponytail: 文本层是长乱码时不会识别；正文里夹的图也不识别。
_SCAN_CHARS_PER_PAGE = 40
_ocr_engine = None
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_PDF_HEADING = re.compile(r"^(第.+[章节条]|[一二三四五六七八九十]+、|\d+(?:\.\d+)*\s+\S.{0,30})$")


class Block(TypedDict, total=False):
    kind: str
    text: str
    page: int | None
    path: str
    headers: list[str]
    n_rows: int
    caption: str


def chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    # 只压段落空白。表格和代码不要走这里。
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks


def parse_document(name: str, raw: bytes) -> list[Block]:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        blocks = _parse_pdf(raw)
    elif suffix == ".md":
        blocks = _parse_markdown(_decode_text(raw))
    elif suffix == ".txt":
        blocks = _parse_plain(_decode_text(raw))
    else:
        raise ValueError("只支持 .txt .md .pdf")

    if not any((b.get("text") or "").strip() for b in blocks):
        raise ValueError("没有可提取的文本")
    return blocks


def read_file(name: str, raw: bytes) -> str:
    # 只要拼起来的正文时用；入库走 parse_document。
    return "\n\n".join(b["text"] for b in parse_document(name, raw) if b.get("text"))


def chunk_blocks(blocks: list[Block]) -> list[str]:
    chunks: list[str] = []
    for b in blocks:
        kind = b.get("kind") or "paragraph"
        if kind == "heading":
            continue
        body = (b.get("text") or "").strip()
        if not body:
            continue
        prefix = _prefix(b)
        if kind == "table":
            summary = table_summary(
                b.get("headers") or [],
                int(b.get("n_rows") or 0),
                b.get("caption") or "",
                b.get("path") or "",
            )
            chunks.extend(_chunk_table(prefix, summary, body))
            continue
        if kind in {"image", "footnote", "code"}:
            chunks.append(f"{prefix}\n{body}" if prefix else body)
            continue
        headed = f"{prefix}\n{body}" if prefix else body
        if len(body) <= _CHUNK_SIZE:
            chunks.append(headed)
            continue
        for part in chunk_text(body):
            chunks.append(f"{prefix}\n{part}" if prefix else part)
    return chunks


def _prefix(block: Block) -> str:
    page = block.get("page")
    path = (block.get("path") or "").strip()
    loc = f"p.{page}" if page else ""
    if loc and path:
        return f"[{loc} / {path}]"
    if loc:
        return f"[{loc}]"
    if path:
        return f"[{path}]"
    return ""


def _chunk_table(prefix: str, summary: str, markdown: str) -> list[str]:
    # 表太长按行切，每块重复表头，避免后半段丢掉列名。
    head = "\n".join(x for x in (prefix, summary) if x)
    lines = markdown.splitlines()
    if len(markdown) <= _CHUNK_SIZE * 2 or len(lines) <= 4:
        return [f"{head}\n{markdown}" if head else markdown]

    sep = lines[1] if len(lines) > 1 else "| --- |"
    header_line = lines[0]
    data = lines[2:]
    out = []
    buf: list[str] = []
    used = 0
    limit = _CHUNK_SIZE
    for row in data:
        extra = len(row) + 1
        if buf and used + extra > limit:
            piece = "\n".join([header_line, sep, *buf])
            out.append(f"{head}\n{piece}" if head else piece)
            buf, used = [], 0
        buf.append(row)
        used += extra
    if buf:
        piece = "\n".join([header_line, sep, *buf])
        out.append(f"{head}\n{piece}" if head else piece)
    return out or [f"{head}\n{markdown}" if head else markdown]


def _parse_plain(text: str) -> list[Block]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    return [{"kind": "paragraph", "text": p, "page": None, "path": ""} for p in parts]


def _parse_markdown(text: str) -> list[Block]:
    blocks: list[Block] = []
    path: list[str] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("```"):
            buf = [line]
            i += 1
            while i < n:
                buf.append(lines[i])
                i += 1
                if lines[i - 1].startswith("```"):
                    break
            blocks.append({"kind": "code", "text": "\n".join(buf), "page": None, "path": " / ".join(path)})
            continue
        if _is_md_table(lines, i):
            table, i = _take_md_table(lines, i)
            blocks.append(_table_block(table, None, " / ".join(path), ""))
            continue
        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            path = path[: level - 1] + [title]
            blocks.append({"kind": "heading", "text": title, "page": None, "path": " / ".join(path)})
            i += 1
            continue
        img = _MD_IMAGE.search(line)
        if img and line.strip().startswith("!["):
            caption = img.group(1).strip() or "图片"
            blocks.append({"kind": "image", "text": f"图片：{caption}", "page": None, "path": " / ".join(path)})
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        para = [line]
        i += 1
        while i < n and lines[i].strip() and not _HEADING.match(lines[i]) and not lines[i].startswith("```") and not _is_md_table(lines, i):
            if _MD_IMAGE.search(lines[i]) and lines[i].strip().startswith("!["):
                break
            para.append(lines[i])
            i += 1
        blocks.append({"kind": "paragraph", "text": "\n".join(para).strip(), "page": None, "path": " / ".join(path)})
    return blocks


def _is_md_table(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines):
        return False
    a, b = lines[i].strip(), lines[i + 1].strip()
    return a.startswith("|") and a.endswith("|") and re.match(r"^\|?\s*:?-{3,}", b) is not None


def _take_md_table(lines: list[str], i: int) -> tuple[list[list[str]], int]:
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip()
        if re.match(r"^\|?\s*:?-{3,}", raw):
            i += 1
            continue
        cells = [c.strip() for c in raw.strip("|").split("|")]
        rows.append(cells)
        i += 1
    return rows, i


def _table_block(rows: list[list[str]], page: int | None, path: str, caption: str) -> Block:
    headers, data = flatten_headers(rows)
    return {
        "kind": "table",
        "text": rows_to_markdown(headers, data),
        "page": page,
        "path": path,
        "headers": headers,
        "n_rows": len(data),
        "caption": caption,
    }


def _parse_pdf(raw: bytes) -> list[Block]:
    try:
        pages = PdfReader(BytesIO(raw)).pages
        parts = [(p.extract_text() or "").strip() for p in pages]
    except Exception as e:
        raise ValueError("无法解析 PDF") from e

    chars = sum(len("".join(t.split())) for t in parts)
    if parts and chars < _SCAN_CHARS_PER_PAGE * len(parts):
        scanned = _ocr_pages(raw)
        scanned_chars = sum(len("".join(w["text"].split())) for page in scanned for w in page)
        if scanned_chars > chars:
            return _blocks_from_ocr(scanned)

    laid = _pdf_layout(raw)
    if laid:
        return laid
    return [{"kind": "paragraph", "text": t, "page": i + 1, "path": ""} for i, t in enumerate(parts) if t]


def _pdf_layout(raw: bytes) -> list[Block] | None:
    try:
        import pdfplumber
    except ImportError:
        return None

    try:
        pdf = pdfplumber.open(BytesIO(raw))
    except Exception:
        return None

    try:
        page_lines: list[list[str]] = []
        page_payloads = []
        for page in pdf.pages:
            words = [
                {"text": w["text"], "x0": w["x0"], "x1": w["x1"], "top": w["top"], "bottom": w["bottom"]}
                for w in page.extract_words() or []
            ]
            tables = _page_tables(page, words)
            images = _page_images(page, words)
            used = [t["bbox"] for t in tables] + [im["bbox"] for im in images]
            leftover = [w for w in words if not _inside(w, used)]
            page_lines.append(reading_lines(leftover, float(page.width or 0)))
            page_payloads.append((tables, images, leftover, float(page.width or 0), float(page.height or 0)))
        junk = repeated_edges(page_lines)
        return _assemble_pdf(page_payloads, junk)
    except Exception:
        return None
    finally:
        pdf.close()


def _page_tables(page, words: list[dict]) -> list[dict]:
    found = []
    try:
        items = page.find_tables() or []
    except Exception:
        items = []
    for t in items:
        try:
            rows = t.extract() or []
            bbox = tuple(t.bbox)
        except Exception:
            continue
        clean = [[(c or "").strip() for c in row] for row in rows if any((c or "").strip() for c in row)]
        if not clean:
            continue
        found.append({"rows": clean, "bbox": bbox, "caption": _caption_above(words, bbox)})
    return found


def _page_images(page, words: list[dict]) -> list[dict]:
    out = []
    for im in page.images or []:
        bbox = (
            float(im.get("x0", 0)),
            float(im.get("top", im.get("y0", 0))),
            float(im.get("x1", 0)),
            float(im.get("bottom", im.get("y1", 0))),
        )
        cap = _caption_below(words, bbox) or _caption_above(words, bbox)
        out.append({"bbox": bbox, "caption": cap})
    return out


def _assemble_pdf(page_payloads: list, junk: set[str]) -> list[Block]:
    blocks: list[Block] = []
    path: list[str] = []
    pending_table: dict | None = None
    pending_page = 0
    for i, (tables, images, leftover, width, height) in enumerate(page_payloads):
        page_no = i + 1
        items: list[tuple[float, str, dict]] = []
        for t in tables:
            items.append((float(t["bbox"][1]), "table", t))
        for im in images:
            items.append((float(im["bbox"][1]), "image", im))
        lines = drop_repeated(reading_lines(leftover, width), junk)
        y_map = _line_tops(leftover)
        for ln in lines:
            items.append((y_map.get(ln, 0.0), "text", {"text": ln, "height": height}))
        items.sort(key=lambda x: x[0])

        if pending_table is not None:
            first = next((it for it in items if it[1] == "table"), None)
            if first and tables_continue(pending_table["rows"], first[2]["rows"], first[2].get("caption") or ""):
                pending_table["rows"] = merge_tables(pending_table["rows"], first[2]["rows"])
                items = [it for it in items if it is not first]
            else:
                blocks.append(_table_block(pending_table["rows"], pending_page, " / ".join(path), pending_table.get("caption") or ""))
                pending_table = None

        for _, kind, payload in items:
            if kind == "table":
                if pending_table is not None:
                    blocks.append(_table_block(pending_table["rows"], pending_page, " / ".join(path), pending_table.get("caption") or ""))
                pending_table = payload
                pending_page = page_no
                continue
            if kind == "image":
                cap = payload.get("caption") or "未识别图注"
                bbox = payload["bbox"]
                text = f"图片 第{page_no}页 位置({int(bbox[0])},{int(bbox[1])})：{cap}"
                blocks.append({"kind": "image", "text": text, "page": page_no, "path": " / ".join(path)})
                continue
            line = payload["text"].strip()
            if not line:
                continue
            if _is_footnote(line, y_map.get(line, 0.0), payload["height"]):
                blocks.append({"kind": "footnote", "text": f"脚注：{line}", "page": page_no, "path": " / ".join(path)})
                continue
            if _is_pdf_heading(line):
                path = _push_heading(path, line)
                blocks.append({"kind": "heading", "text": line, "page": page_no, "path": " / ".join(path)})
                continue
            blocks.append({"kind": "paragraph", "text": line, "page": page_no, "path": " / ".join(path)})

    if pending_table is not None:
        blocks.append(_table_block(pending_table["rows"], pending_page, " / ".join(path), pending_table.get("caption") or ""))
    return _merge_paragraphs(blocks)


def _blocks_from_ocr(pages: list[list[dict]]) -> list[Block]:
    page_lines = []
    payloads = []
    for words in pages:
        width = max((w["x1"] for w in words), default=0.0)
        height = max((w["bottom"] for w in words), default=0.0)
        tables = tables_from_words(words)
        used = [t["bbox"] for t in tables]
        leftover = [w for w in words if not _inside(w, used)]
        page_lines.append(reading_lines(leftover, width))
        payloads.append(([{"rows": t["rows"], "bbox": t["bbox"], "caption": ""} for t in tables], [], leftover, width, height))
    junk = repeated_edges(page_lines)
    return _assemble_pdf(payloads, junk)


def _merge_paragraphs(blocks: list[Block]) -> list[Block]:
    # 同一页、同一标题下的短行并成段，避免每行一块冲淡检索。
    out: list[Block] = []
    for b in blocks:
        if out and b.get("kind") == "paragraph" and out[-1].get("kind") == "paragraph" and out[-1].get("page") == b.get("page") and out[-1].get("path") == b.get("path"):
            out[-1]["text"] = f"{out[-1]['text']} {b['text']}"
            continue
        out.append(b)
    return out


def _line_tops(words: list[dict]) -> dict[str, float]:
    tops: dict[str, float] = {}
    for ln in reading_lines(words, max((w["x1"] for w in words), default=0.0) + 1):
        tops.setdefault(ln, 0.0)
    for w in words:
        key = w["text"]
        for ln in list(tops):
            if key in ln and (tops[ln] == 0.0 or w["top"] < tops[ln]):
                tops[ln] = w["top"]
    return tops


def _caption_above(words: list[dict], bbox: tuple) -> str:
    x0, top, x1, _ = bbox
    near = [w for w in words if w["bottom"] <= top and top - w["bottom"] < 28 and w["x0"] < x1 and w["x1"] > x0]
    return " ".join(w["text"] for w in sorted(near, key=lambda w: (w["top"], w["x0"]))).strip()


def _caption_below(words: list[dict], bbox: tuple) -> str:
    x0, _, x1, bottom = bbox
    near = [w for w in words if w["top"] >= bottom and w["top"] - bottom < 28 and w["x0"] < x1 and w["x1"] > x0]
    return " ".join(w["text"] for w in sorted(near, key=lambda w: (w["top"], w["x0"]))).strip()


def _inside(word: dict, boxes: list[tuple]) -> bool:
    cx = (word["x0"] + word["x1"]) / 2
    cy = (word["top"] + word["bottom"]) / 2
    for x0, top, x1, bottom in boxes:
        if x0 <= cx <= x1 and top <= cy <= bottom:
            return True
    return False


def _is_footnote(line: str, top: float, height: float) -> bool:
    if height and top > height * 0.8 and re.match(r"^([0-9]{1,2}|[*†‡§])\s+", line):
        return True
    return line.startswith("脚注") or line.startswith("注：")


def _is_pdf_heading(line: str) -> bool:
    s = line.strip()
    if len(s) > 40 or len(s) < 2:
        return False
    return _PDF_HEADING.match(s) is not None


def _push_heading(path: list[str], title: str) -> list[str]:
    dots = title.split()[0].count(".") if title[:1].isdigit() else 0
    if title.startswith("第") and title.endswith(("章",)):
        return [title]
    if title.startswith("第") and title.endswith(("节",)):
        return (path[:1] or [title]) + [title]
    if dots:
        return path[:dots] + [title]
    return path[:1] + [title] if path else [title]


def _ocr_pages(raw: bytes) -> list[list[dict]]:
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
        logging.getLogger("rapidocr").setLevel(logging.ERROR)

    try:
        doc = pdfium.PdfDocument(raw)
    except Exception as e:
        raise ValueError("无法识别扫描页") from e

    pages: list[list[dict]] = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            bitmap = page.render(scale=2)
            try:
                result = _ocr_engine(bitmap.to_pil(), text_score=0.5)
            finally:
                bitmap.close()
                page.close()
            txts = list(getattr(result, "txts", None) or [])
            boxes = list(getattr(result, "boxes", None) or [])
            if txts and boxes and len(boxes) == len(txts):
                pages.append(ocr_to_words(txts, boxes))
            elif txts:
                pages.append([{"text": t, "x0": float(j), "x1": float(j + 1), "top": 0.0, "bottom": 1.0} for j, t in enumerate(txts)])
            else:
                pages.append([])
    except Exception as e:
        raise ValueError("无法识别扫描页") from e
    finally:
        doc.close()
    return pages


def _decode_text(raw: bytes) -> str:
    # Windows 中文文件经常是 gbk，utf-8 失败再试一次。
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("文本编码无法识别")
