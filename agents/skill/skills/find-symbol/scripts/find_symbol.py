"""Print path:line for every class/def whose (qualified) name matches the given names."""
import ast
import os
import sys

root = "/workspace" if os.path.isdir("/workspace") else os.getcwd()
wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
SKIP = {".git", "node_modules", ".venv", "build", "dist", "__pycache__"}


def walk(node, prefix, path, out):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            qual = f"{prefix}.{child.name}" if prefix else child.name
            kind = "class" if isinstance(child, ast.ClassDef) else "def"
            out.append((path, child.lineno, kind, qual))
            walk(child, qual, path, out)


found = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in SKIP]
    for name in filenames:
        if not name.endswith(".py"):
            continue
        full = os.path.join(dirpath, name)
        try:
            with open(full, encoding="utf-8", errors="replace") as fh:
                tree = ast.parse(fh.read())
        except (SyntaxError, ValueError, OSError):
            continue
        walk(tree, "", os.path.relpath(full, root), found)

hits = [f for f in found if any(f[3] == w or f[3].endswith("." + w) for w in wanted)]
for path, line, kind, qual in sorted(hits)[:60]:
    print(f"{path}:{line}  {kind}  {qual}")
if not hits:
    print("no definition found for: " + ", ".join(wanted))
