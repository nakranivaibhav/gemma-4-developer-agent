"""Build a code graph in the official competition format from a repo checkout.

Reverse-engineered from the 127 official graphs (data/graphs/<repo>_<commit>.json). Checked by graph/graph_match.py:
all 127 rebuild with the same node ids (367,985), the same node texts and the same edge pairs (226,247), 0 extra,
0 missing (2026-10-17). The quirks are reproduced on purpose, so the graph tools behave on outside repos as they do
on Kaggle.

Nodes
- Every *.py file under the root, tests and docs included.
- Module name: the path from the deepest "source root" above the file, with a trailing `__init__` dropped. Source
  roots are the repo root plus the parent folder of every top-level package (a folder with `__init__.py` whose
  parent has none). So `src/requests/models.py` -> `requests.models`, fastapi's `docs_src/x/y.py` -> `x.y`, and
  `docs/_themes/z.py` (no package) -> `docs._themes.z`.
- One node per `def` and `class`. `async def` is never a node (host confirmed, forum #745998).
- Naming keeps a single "current class", set on entering a class and cleared (not restored) on leaving it. So a
  method is `mod.Class.m`, a function nested in a method is `mod.Class.inner`, a class nested anywhere is `mod.Inner`,
  and every method after a nested class in the same class body loses its class prefix (`mod.m`).
- A repeated id keeps its first position and the last definition's text (networkx add_node).
- Function text = exact source segment (`ast.get_source_segment`, no decorators). Class text = `ast.unparse` of a
  stub: decorators, header, cleaned docstring (or `None`), then each direct method with body `...`.

Edges (type `calls`; each pair stored once per direction with key 0; a self-loop gets keys 0 and 1)
- class <-> member, for each def visited while the current class is set (so the quirk above also drops these).
- Uses, made by the innermost enclosing def (class-body code and class decorators count for the def around the class,
  if any, never for the class itself):
  - the callee of every call (`f()`, `mod.f()`, `a.b.C()`), including calls in the def's decorators and arguments
    (defaults, annotations);
  - `except X` / `except a.X` with a single type (tuples are skipped);
  - class bases (made by the class).
- Names resolve through a per-file table filled in source order: imports anywhere in the file (including inside
  functions), class names, and function names that are not methods, each as `mod.name`. A name used before its def
  in the same file does not resolve; a method name never shadows an import. Relative imports resolve against the
  module's parent even in an `__init__.py` (so `from .x import y` in `pkg/__init__.py` points one level too high).
- A use links when its resolved dotted name is a node. For calls and except types through an imported name, if it is
  not a node, it links to the one node with the same last name whose id starts with the resolved prefix (plain
  string prefix), when exactly one exists: `requests.Request` -> `requests.models.Request`, `rich.color.ColorTriplet`
  -> `rich.color_triplet.ColorTriplet`, `Prompt.ask` -> `PromptBase.ask`; `requests.get` has several candidates and
  does not link. Bases never use this fallback. `self.m()` never resolves.

Order
- The official file order is the builder's hashed directory walk and differs per graph; the official edge order fits
  no single file order either (best order breaks 319 of 3,714 constraints on requests_0b401c76). Neither can be
  reproduced. Here: os.walk with each folder listing shuffled by a seed (default: repo name), defs in source order,
  edges in link order. Given the official file order, node order matches on all 127 graphs.
"""
import ast
import copy
import json
import os
import random
from pathlib import Path

import networkx as nx


def source_roots(root: Path, py_files):
    """The parent folder of every top-level package (a folder with __init__.py whose parent has none), plus the repo
    root. A file is named from the deepest root above it."""
    roots = {()}
    for path in py_files:
        if path.name != '__init__.py':
            continue
        d = path.parent.relative_to(root).parts
        if d and not (root.joinpath(*d[:-1]) / '__init__.py').is_file():
            roots.add(d[:-1])
    return roots


def _module(root: Path, path: Path, roots):
    rel = path.relative_to(root)
    dirs = rel.parent.parts
    base = max((r for r in roots if dirs[:len(r)] == r), key=len)
    parts = list(rel.with_suffix('').parts)[len(base):]
    if parts and parts[-1] == '__init__':
        parts = parts[:-1]
    # relative imports resolve against the module's parent even in an __init__.py (official quirk)
    return '.'.join(parts), parts[:-1]


def _class_text(node: ast.ClassDef) -> str:
    stub = copy.deepcopy(node)
    body = [ast.Expr(ast.Constant(ast.get_docstring(node)))]
    for child in node.body:
        if isinstance(child, ast.FunctionDef):
            f = copy.deepcopy(child)
            f.body = [ast.Expr(ast.Constant(...))]
            body.append(f)
    stub.body = body
    return ast.unparse(stub)


def _dotted(expr):
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return parts[::-1]
    return None


def walk_py(root: Path, seed=None):
    """*.py paths in os.walk order (a folder's files, then its subfolders). With a seed, each listing is shuffled,
    standing in for the hashed directory order the official builder got; without one, it is sorted."""
    rng = random.Random(seed)
    out = []
    for d, subdirs, names in os.walk(root):
        subdirs[:] = sorted(x for x in subdirs if x != '.git')
        names = sorted(n for n in names if n.endswith('.py'))
        if seed is not None:
            rng.shuffle(subdirs)
            rng.shuffle(names)
        out += [Path(d) / n for n in names if (Path(d) / n).is_file()]
    return out


def parse_repo(root: Path, seed=None, module_order=None):
    """[(module, package_parts, source, tree)] for every parseable .py file, in walk order (or sorted by
    `module_order`, a list of module names, for checks against an official file). Also returns skipped files."""
    files, skipped = [], []
    paths = walk_py(root, seed)
    roots = source_roots(root, paths)
    for path in paths:
        try:
            src = path.read_text(encoding='utf-8')
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            skipped.append((str(path.relative_to(root)), type(e).__name__))
            continue
        mod, pkg = _module(root, path, roots)
        files.append((mod, pkg, src, tree))
    if module_order is not None:
        rank = {m: i for i, m in enumerate(module_order)}
        files.sort(key=lambda f: rank.get(f[0], len(rank)))
    return files, skipped


def build(root: Path, seed=None, module_order=None):
    """Return (nodes {id: text} in insertion order, links [(a, b)] in link order, skipped files)."""
    root = Path(root)
    files, skipped = parse_repo(root, seed, module_order)
    nodes = {}

    for mod, _pkg, src, tree in files:
        class Collect(ast.NodeVisitor):
            cur = None

            def visit_ClassDef(self, n):
                nodes[f'{mod}.{n.name}'] = _class_text(n)  # a repeat keeps its first position (networkx add_node)
                self.cur = n.name
                self.generic_visit(n)
                self.cur = None

            def visit_FunctionDef(self, n):
                q = f'{mod}.{self.cur}.{n.name}' if self.cur else f'{mod}.{n.name}'
                nodes[q] = ast.get_source_segment(src, n)
                self.generic_visit(n)

        Collect().visit(tree)

    by_short = {}
    for q in nodes:
        by_short.setdefault(q.rsplit('.', 1)[1], []).append(q)

    links = []

    def link(a, b):
        if a in nodes and b in nodes:
            links.append((a, b))

    for mod, pkg, _src, tree in files:
        table = {}  # local name -> (dotted target, 'import' | 'def')

        class Link(ast.NodeVisitor):
            cur = None
            owner = []
            in_class = False

            def resolve(self, expr, fuzzy=False):
                dp = _dotted(expr)
                if not dp or dp[0] not in table:
                    return None
                target, origin = table[dp[0]]
                full = '.'.join([target] + dp[1:])
                if full in nodes:
                    return full
                if fuzzy and origin == 'import':
                    head = full.rpartition('.')[0]
                    cands = [x for x in by_short.get(dp[-1], ()) if x.startswith(head)]
                    if len(cands) == 1:
                        return cands[0]
                return None

            def use(self, expr, fuzzy=False):
                if self.owner:
                    t = self.resolve(expr, fuzzy)
                    if t:
                        link(self.owner[-1], t)

            def visit_Import(self, n):
                for a in n.names:
                    if a.asname:
                        table[a.asname] = (a.name, 'import')
                    else:
                        top = a.name.split('.')[0]
                        table[top] = (top, 'import')

            def visit_ImportFrom(self, n):
                if n.level:
                    base = pkg[:len(pkg) - (n.level - 1)] + ([n.module] if n.module else [])
                else:
                    base = [n.module] if n.module else []
                for a in n.names:
                    table[a.asname or a.name] = ('.'.join(base + [a.name]), 'import')

            def visit_ClassDef(self, n):
                q = f'{mod}.{n.name}'
                table[n.name] = (q, 'def')
                # class decorators and class-body code count for the enclosing def, if any (not the class)
                in_class, self.in_class = self.in_class, True
                for d in n.decorator_list:
                    self.visit(d)
                for b in n.bases:
                    t = self.resolve(b)
                    if t:
                        link(q, t)
                self.cur = n.name
                for st in n.body:
                    self.visit(st)
                self.cur = None
                self.in_class = in_class

            def visit_FunctionDef(self, n):
                q = f'{mod}.{self.cur}.{n.name}' if self.cur else f'{mod}.{n.name}'
                if self.cur:
                    link(f'{mod}.{self.cur}', q)
                if not self.in_class:  # a method does not enter the name table
                    table[n.name] = (f'{mod}.{n.name}', 'def')
                in_class, self.in_class = self.in_class, False
                self.owner.append(q)
                for d in n.decorator_list:
                    self.visit(d)
                self.visit(n.args)
                for st in n.body:
                    self.visit(st)
                self.owner.pop()
                self.in_class = in_class

            def visit_AsyncFunctionDef(self, n):
                # not a node; its body still runs under the enclosing owner
                self.generic_visit(n)

            def visit_Call(self, n):
                self.use(n.func, fuzzy=True)
                self.generic_visit(n)

            def visit_ExceptHandler(self, n):
                if n.type is not None and not isinstance(n.type, ast.Tuple):
                    self.use(n.type, fuzzy=True)
                self.generic_visit(n)

        Link().visit(tree)
    return nodes, links, skipped


def pairs(links):
    """Undirected edge set, for comparisons (a self-loop is a one-element set)."""
    return {frozenset(p) for p in links}


def to_graph(repo_name: str, nodes: dict, links: list) -> nx.MultiDiGraph:
    """The official way: add_node per def in visit order, then for each link not yet present, add it both ways (so
    every pair is stored in both directions with key 0, and a self-loop gets keys 0 and 1)."""
    g = nx.MultiDiGraph(repo_name=repo_name)
    for q, text in nodes.items():
        g.add_node(q, name=q, text=text)
    for a, b in links:
        if not g.has_edge(a, b):
            g.add_edge(a, b, type='calls')
            g.add_edge(b, a, type='calls')
    return g


def to_json(g: nx.MultiDiGraph) -> str:
    """Same serialisation as swegemma.graph.save_graph_to_json."""
    return json.dumps(nx.node_link_data(g, edges='edges'), indent=2)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Build an official-format code graph from a repo checkout.')
    ap.add_argument('root')
    ap.add_argument('repo_name', help='e.g. requests_<commit>; also the output file stem')
    ap.add_argument('--out', default='.')
    ap.add_argument('--seed', default=None, help='shuffle folder listings with this seed (default: the repo_name)')
    x = ap.parse_args()
    n, links, skipped = build(Path(x.root), seed=x.seed or x.repo_name)
    g = to_graph(x.repo_name, n, links)
    path = Path(x.out) / f'{x.repo_name}.json'
    path.write_text(to_json(g))
    print(f'{path}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, {len(skipped)} files skipped')
