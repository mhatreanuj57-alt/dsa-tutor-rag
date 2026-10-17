import re
from pathlib import Path

MAX_CHARS = 1200
OVERLAP = 150


def _split_functions(text: str):
    lines = text.splitlines(keepends=True)
    chunks, current, depth = [], [], 0
    started = False
    for line in lines:
        stripped = line.strip()
        current.append(line)
        opens = line.count("{")
        closes = line.count("}")
        if opens and not started:
            started = True
        depth += opens - closes
        joined = "".join(current)
        if started and depth <= 0 and len(joined) > 200:
            chunks.append(joined)
            current, depth, started = [], 0, False
        elif len(joined) > MAX_CHARS:
            chunks.append(joined)
            current, depth, started = [], 0, False
    if current:
        chunks.append("".join(current))
    return [c for c in chunks if c.strip()]


def _split_markdown(text: str):
    parts = re.split(r"\n(?=#{1,4}\s)", text)
    out = []
    for part in parts:
        if len(part) <= MAX_CHARS:
            out.append(part)
            continue
        words = part.split(" ")
        buf = []
        for w in words:
            buf.append(w)
            if sum(len(x) + 1 for x in buf) > MAX_CHARS:
                out.append(" ".join(buf))
                buf = buf[-OVERLAP // 10:]
        if buf:
            out.append(" ".join(buf))
    return [p for p in out if p.strip()]


def chunk_file(path: Path, root: Path):
    rel = path.relative_to(root).as_posix()
    topic = rel.split("/")[0] if "/" in rel else "root"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if path.suffix == ".cpp":
        pieces = _split_functions(text)
    elif path.suffix == ".md":
        pieces = _split_markdown(text)
    else:
        pieces = [text[i:i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS - OVERLAP)]

    chunks = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if len(piece) < 80:
            continue
        first_line = next((l.strip("# /") for l in piece.splitlines() if l.strip()), "")[:120]
        chunks.append({
            "text": f"[{topic}] file: {rel} | {first_line}\n\n{piece}",
            "source": rel,
            "topic": topic,
            "chunk_index": i,
        })
    return chunks


def chunk_repo(root: Path):
    root = Path(root)
    all_chunks = []
    for pattern in ("**/*.cpp", "**/*.md", "**/*.txt"):
        for path in sorted(root.glob(pattern)):
            if ".git" in path.parts:
                continue
            all_chunks.extend(chunk_file(path, root))
    return all_chunks
