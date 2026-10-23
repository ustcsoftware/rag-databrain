import re
from collections import Counter

# 空单元格按合并格补全；跨页表按列数和表头判断是否续上。


def fill_merged(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return []
    width = max(len(r) for r in rows)
    out = [[(c or "").strip() for c in r] + [""] * (width - len(r)) for r in rows]
    for r in range(len(out)):
        for c in range(width):
            if not out[r][c] and r and out[r - 1][c]:
                out[r][c] = out[r - 1][c]
    for r in range(len(out)):
        for c in range(width):
            if not out[r][c] and c and out[r][c - 1]:
                out[r][c] = out[r][c - 1]
    return out


def flatten_headers(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    # 先在未补全的表上认多级表头，再补合并格，竖着拼成「部门 / 销售」。
    if not rows:
        return [], []
    n = 1
    while n < min(3, len(rows)) and _headerish(rows[n]) and _sparser_than_later(rows, n):
        n += 1
    heads = fill_merged(rows[:n])
    width = len(heads[0])
    labels = []
    for c in range(width):
        parts = []
        for row in heads:
            v = (row[c] if c < len(row) else "").strip()
            if v and v not in parts:
                parts.append(v)
        labels.append(" / ".join(parts) if parts else f"列{c + 1}")
    return labels, fill_merged(rows[n:])


def tables_continue(prev: list[list[str]], nxt: list[list[str]], caption: str = "") -> bool:
    if caption.strip() or not prev or not nxt:
        return False
    if len(prev[0]) != len(nxt[0]):
        return False
    return _row_eq(prev[0], nxt[0]) or not _headerish(nxt[0])


def merge_tables(prev: list[list[str]], nxt: list[list[str]]) -> list[list[str]]:
    if _row_eq(prev[0], nxt[0]):
        return prev + nxt[1:]
    return prev + nxt


def rows_to_markdown(headers: list[str], data: list[list[str]]) -> str:
    def cell(s: str) -> str:
        return (s or "").replace("|", "\\|").replace("\n", " ").strip()

    cols = max(len(headers), 1)
    heads = [cell(h) for h in (headers or [f"列{i + 1}" for i in range(cols)])]
    lines = [
        "| " + " | ".join(heads) + " |",
        "| " + " | ".join("---" for _ in heads) + " |",
    ]
    for row in data:
        padded = [cell(row[i]) if i < len(row) else "" for i in range(len(heads))]
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


def table_summary(headers: list[str], n_rows: int, caption: str = "", path: str = "") -> str:
    title = caption.strip() or path.strip() or "表格"
    cols = "、".join(h for h in headers if h) or "未识别列"
    return f"表格：{title}，列：{cols}，共 {n_rows} 行。"


def words_to_lines(words: list[dict]) -> list[str]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w["top"], w["x0"]))
    groups: list[list[dict]] = [[ordered[0]]]
    for w in ordered[1:]:
        prev = groups[-1][-1]
        band = max(3.0, 0.5 * max(prev["bottom"] - prev["top"], 1))
        if abs(w["top"] - prev["top"]) <= band:
            groups[-1].append(w)
        else:
            groups.append([w])
    return [" ".join(x["text"] for x in sorted(g, key=lambda x: x["x0"])) for g in groups]


def split_columns(words: list[dict], page_width: float) -> list[list[dict]]:
    # 中间有明显沟才分栏，避免单栏被中线切开。
    if len(words) < 16 or page_width <= 0:
        return [words]
    mid = page_width / 2
    left = [w for w in words if (w["x0"] + w["x1"]) / 2 < mid]
    right = [w for w in words if (w["x0"] + w["x1"]) / 2 >= mid]
    if len(left) < 8 or len(right) < 8:
        return [words]
    gutter = min(w["x0"] for w in right) - max(w["x1"] for w in left)
    if gutter < 8:
        return [words]
    return [left, right]


def reading_lines(words: list[dict], page_width: float) -> list[str]:
    lines: list[str] = []
    for col in split_columns(words, page_width):
        lines.extend(words_to_lines(col))
    return lines


def repeated_edges(pages: list[list[str]]) -> set[str]:
    # 多页顶端/底端重复的，当页眉页脚丢掉。
    counts: Counter[str] = Counter()
    for lines in pages:
        for raw in lines[:2] + lines[-2:]:
            key = " ".join(raw.split())
            if key:
                counts[key] += 1
    n = len(pages)
    if n < 2:
        return set()
    need = max(2, (n + 1) // 2)
    return {s for s, k in counts.items() if k >= need}


def drop_repeated(lines: list[str], junk: set[str]) -> list[str]:
    return [ln for ln in lines if " ".join(ln.split()) not in junk]


def ocr_to_words(txts: list[str], boxes: list) -> list[dict]:
    words = []
    for text, box in zip(txts, boxes):
        t = (text or "").strip()
        if not t:
            continue
        x0, y0, x1, y1 = _box_xy(box)
        words.append({"text": t, "x0": x0, "x1": x1, "top": y0, "bottom": y1})
    return words


def tables_from_words(words: list[dict]) -> list[dict]:
    # 多行共享同一组 x 槽，当成表，避免 OCR 结果只剩空格拼接。
    if len(words) < 6:
        return []
    lines = _word_rows(words)
    if len(lines) < 3:
        return []
    slots = _column_slots(lines)
    if len(slots) < 3:
        return []
    start = None
    tables = []
    for i, row in enumerate(lines):
        aligned = len(row) >= 3 and _row_fits_slots(row, slots)
        if aligned and start is None:
            start = i
        elif not aligned and start is not None:
            if i - start >= 3:
                tables.append(_rows_bbox(lines[start:i]))
            start = None
    if start is not None and len(lines) - start >= 3:
        tables.append(_rows_bbox(lines[start:]))
    return tables


def _box_xy(box) -> tuple[float, float, float, float]:
    if box is None:
        return 0.0, 0.0, 0.0, 0.0
    if len(box) >= 8 and not isinstance(box[0], (list, tuple)):
        xs = [float(box[i]) for i in range(0, 8, 2)]
        ys = [float(box[i]) for i in range(1, 8, 2)]
        return min(xs), min(ys), max(xs), max(ys)
    if len(box) == 4 and not isinstance(box[0], (list, tuple)):
        x0, y0, x1, y1 = (float(v) for v in box[:4])
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    pts = [(float(p[0]), float(p[1])) for p in box]
    return min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)


def _word_rows(words: list[dict]) -> list[list[dict]]:
    ordered = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows: list[list[dict]] = [[ordered[0]]]
    for w in ordered[1:]:
        prev = rows[-1][-1]
        band = max(3.0, 0.5 * max(prev["bottom"] - prev["top"], 1))
        if abs(w["top"] - prev["top"]) <= band:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w["x0"]) for r in rows]


def _column_slots(lines: list[list[dict]]) -> list[float]:
    centers = []
    for row in lines:
        if len(row) < 3:
            continue
        centers.extend((w["x0"] + w["x1"]) / 2 for w in row)
    if not centers:
        return []
    centers.sort()
    slots = [centers[0]]
    for x in centers[1:]:
        if x - slots[-1] > 18:
            slots.append(x)
        else:
            slots[-1] = (slots[-1] + x) / 2
    return slots if len(slots) >= 3 else []


def _row_fits_slots(row: list[dict], slots: list[float]) -> bool:
    hits = 0
    for w in row:
        c = (w["x0"] + w["x1"]) / 2
        if any(abs(c - s) < 20 for s in slots):
            hits += 1
    return hits >= min(3, len(slots))


def _rows_bbox(rows: list[list[dict]]) -> dict:
    cells = []
    flat = []
    for row in rows:
        cells.append([w["text"] for w in row])
        flat.extend(row)
    return {
        "rows": cells,
        "bbox": (
            min(w["x0"] for w in flat),
            min(w["top"] for w in flat),
            max(w["x1"] for w in flat),
            max(w["bottom"] for w in flat),
        ),
    }


def _headerish(row: list[str]) -> bool:
    cells = [(c or "").strip() for c in row]
    nonempty = [c for c in cells if c]
    if not nonempty:
        return False
    nums = sum(1 for c in nonempty if _is_number(c))
    return nums * 2 < len(nonempty)


def _sparser_than_later(rows: list[list[str]], idx: int) -> bool:
    empty = sum(1 for c in rows[idx] if not (c or "").strip())
    later = rows[idx + 1 :]
    if empty == 0 or not later:
        return False
    avg = sum(sum(1 for c in r if not (c or "").strip()) for r in later) / len(later)
    return empty >= avg


def _row_eq(a: list[str], b: list[str]) -> bool:
    if len(a) != len(b):
        return False
    return all((x or "").strip() == (y or "").strip() for x, y in zip(a, b))


def _is_number(s: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?%?", s.strip()))
