"""Search the repository's functions and classes by words or by name. Standard library only.

Prints the best matches as `path:line  kind  qualified.name` plus the definition line, best first.
Words are matched against each definition's name (worth more) and its code; rare words count more than common
ones; test code ranks below source code. A query that is a symbol name (`Class.method`, `func_name`) puts that
definition first.
"""
import ast
import math
import os
import re
import sys

ROOT = "/workspace" if os.path.isdir("/workspace") else os.getcwd()
SKIP = {".git", ".hg", "node_modules", ".venv", "venv", "env", "build", "dist", "__pycache__", "site-packages",
        ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".eggs"}
STOP = {"the", "and", "for", "not", "with", "when", "from", "into", "that", "this", "are", "was", "how", "use",
        "should", "does", "can", "a", "an", "of", "to", "in", "on", "is", "it", "be", "or", "by", "as", "at", "if"}


def split_words(text):
    """`UploadFile.max_size` -> upload, file, max, size (lowercase, 2+ characters, no stop words)."""
    out = []
    for part in re.findall(r"[A-Za-z0-9]+", text):
        for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", part):
            w = w.lower()
            if len(w) >= 2 and w not in STOP:
                out.append(w)
    return out


def parse_args(argv):
    words, k = [], 10
    i = 0
    while i < len(argv):
        a = str(argv[i])
        if a in ("--k", "-k", "--top", "-n") and i + 1 < len(argv):
            try:
                k = max(1, min(30, int(argv[i + 1])))
            except ValueError:
                pass
            i += 2
            continue
        if a.startswith("-") and a.lstrip("-") and not a.lstrip("-")[0].isdigit():
            i += 1  # a key such as --query: its value follows
            continue
        words.extend(a.split())
        i += 1
    return words, k


def module_name(rel):
    mod = rel[:-3].replace(os.sep, ".")
    return mod[: -len(".__init__")] if mod.endswith(".__init__") else mod


def collect():
    defs = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP and not d.endswith(".egg-info"))
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, ROOT)
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
                tree = ast.parse(src)
            except (SyntaxError, ValueError, OSError, RecursionError):
                continue
            lines = src.splitlines()
            stack = [(tree, module_name(rel))]
            while stack:
                node, prefix = stack.pop()
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        qual = prefix + "." + child.name
                        end = getattr(child, "end_lineno", child.lineno) or child.lineno
                        body = "\n".join(lines[child.lineno - 1:end])
                        kind = "class" if isinstance(child, ast.ClassDef) else "def"
                        head = lines[child.lineno - 1].strip() if child.lineno <= len(lines) else ""
                        defs.append((rel, child.lineno, kind, qual, head, body))
                        stack.append((child, qual))
                    elif not isinstance(child, (ast.expr, ast.Import, ast.ImportFrom)):
                        stack.append((child, prefix))
    return defs


def is_test(rel, qual):
    low = rel.lower()
    return (low.startswith(("test", "tests" + os.sep)) or (os.sep + "tests" + os.sep) in low
            or os.path.basename(low).startswith("test_") or ".test_" in qual.lower())


def is_example(rel):
    low = rel.lower()
    return low.startswith(("docs", "docs_src", "examples", "example")) or (os.sep + "examples" + os.sep) in low


def main():
    words, k = parse_args(sys.argv[1:])
    if not words:
        print('usage: args={"query": "two to four words"} or args={"query": "Class.method"}')
        return
    query = " ".join(words)
    qwords = list(dict.fromkeys(split_words(query)))
    defs = collect()
    if not defs:
        print("no Python definitions found under " + ROOT)
        return

    name_words = [set(split_words(d[3].split(".")[-1])) | set(split_words(d[3])) for d in defs]
    body_low = [d[5].lower() for d in defs]
    n = len(defs)
    idf = {}
    for w in qwords:
        df = sum(1 for i in range(n) if w in name_words[i] or w in body_low[i])
        idf[w] = math.log(1.0 + n / (1.0 + df))

    # A symbol name (no spaces) matches the qualified name exactly or as a dotted suffix.
    exact = set()
    if len(words) == 1 and re.fullmatch(r"[A-Za-z_][\w.]*", words[0]):
        target = words[0].lower()
        for i, d in enumerate(defs):
            ql = d[3].lower()
            if ql == target or ql.endswith("." + target):
                exact.add(i)

    scored = []
    for i, d in enumerate(defs):
        s = 0.0
        for w in qwords:
            if w in name_words[i]:
                s += 3.0 * idf[w]
            if w in body_low[i]:
                s += 1.0 * idf[w]
        if i in exact:
            s += 1000.0
        if s <= 0:
            continue
        if is_test(d[0], d[3]):
            s *= 0.5
        elif is_example(d[0]):
            s *= 0.7
        scored.append((-s, d[0], d[1], i))
    scored.sort()
    if not scored:
        print("no match for: " + query + ". Try other words, or grep.")
        return
    # Copies of one name in one folder (tutorial variants) show once, with a count.
    top, groups = [], {}
    for item in scored:
        d = defs[item[3]]
        key = (os.path.dirname(d[0]), d[3].split(".")[-1])
        if key in groups:
            groups[key][1] += 1
            continue
        if len(top) == k:
            continue
        groups[key] = [item, 0]
        top.append(key)
    best = -scored[0][0]
    print("query: %s  (%d definitions searched, best %d shown)" % (query, n, len(top)))
    for key in top:
        (neg, rel, line, i), more = groups[key]
        d = defs[i]
        extra = "  (+%d more like it in %s/)" % (more, key[0]) if more else ""
        print("%s:%d  %s  %s  score=%.2f%s" % (rel, line, d[2], d[3], -neg / best, extra))
        print("    " + d[4][:160])


if __name__ == "__main__":
    main()
