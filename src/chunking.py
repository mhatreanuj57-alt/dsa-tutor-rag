import ast
import os
import re
from pathlib import Path

MAX_CHARS = 1200
OVERLAP = 150
CODE_LANGUAGES = {"cpp": "C++", "python": "Python", "java": "Java"}
FILE_LANGUAGES = {".cpp": "cpp", ".py": "python", ".java": "java", ".md": "markdown", ".txt": "text"}
IGNORED_DIRS = {"__pycache__", "out", "node_modules", "venv", "env", "build", "dist"}


def _split_lines(text: str):
    # shortcut: a single long line may exceed MAX_CHARS; use token splitting for minified inputs.
    chunks, current = [], []
    for line in text.splitlines(keepends=True):
        if current and len("".join(current)) + len(line) > MAX_CHARS:
            chunks.append("".join(current))
            tail, size = [], 0
            for previous in reversed(current):
                if size + len(previous) > OVERLAP:
                    break
                tail.insert(0, previous)
                size += len(previous)
            current = tail
        current.append(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _split_python(text: str):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _split_lines(text)
    lines = text.splitlines(keepends=True)
    starts = {0, len(lines)}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            starts.add(min([node.lineno, *(d.lineno for d in node.decorator_list)]) - 1)
    starts = sorted(starts)
    return [piece for start, end in zip(starts, starts[1:]) for piece in _split_lines("".join(lines[start:end]))]


def _split_functions(text: str):
    # shortcut: braces inside comments/strings affect boundaries; use a parser if exact boundaries become necessary.
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
                buf = buf[-OVERLAP // 10 :]
        if buf:
            out.append(" ".join(buf))
    return [p for p in out if p.strip()]


def chunk_file(path: Path, root: Path, language=None, source_prefix=None):
    rel = path.relative_to(root).as_posix()
    parts = path.relative_to(root).parts
    if len(parts) > 2 and parts[0] == "src":
        parts = parts[1:]
    topic = parts[0] if len(parts) > 1 else "root"
    language = language or FILE_LANGUAGES.get(path.suffix.lower(), "text")
    source = f"{source_prefix}/{rel}" if source_prefix else rel
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as error:
        raise RuntimeError(f"Cannot read source file: {path}") from error

    if path.suffix.lower() in (".cpp", ".java"):
        pieces = _split_functions(text)
    elif path.suffix.lower() == ".py":
        pieces = _split_python(text)
    elif path.suffix.lower() == ".md":
        pieces = _split_markdown(text)
    else:
        pieces = _split_lines(text)

    chunks = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if len(piece) < 80:
            continue
        first_line = next((l.strip("# /") for l in piece.splitlines() if l.strip()), "")[:120]
        chunks.append(
            {
                "text": f"[{language} | {topic}] file: {source} | {first_line}\n\n{piece}",
                "source": source,
                "topic": topic,
                "language": language,
                "chunk_index": i,
            }
        )
    return chunks


def chunk_repo(root: Path, language=None, source_prefix=None):
    root = Path(root)
    all_chunks = []

    def read_error(error):
        raise RuntimeError(f"Cannot read notes directory: {error.filename}") from error

    for directory, dirs, files in os.walk(root, onerror=read_error):
        dirs[:] = sorted(name for name in dirs if not name.startswith(".") and name not in IGNORED_DIRS)
        for filename in sorted(files):
            path = Path(directory) / filename
            file_language = FILE_LANGUAGES.get(path.suffix.lower())
            if file_language is None or filename == "verify_collection.py":
                continue
            if language and file_language in CODE_LANGUAGES and file_language != language:
                continue
            all_chunks.extend(chunk_file(path, root, language=language, source_prefix=source_prefix))
    return all_chunks
