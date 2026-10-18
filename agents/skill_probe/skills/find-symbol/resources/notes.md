# Code graph notes

- Graph links have no direction: a caller and a callee look the same.
- Calls made through `self` or another object are not linked.
- Graph nodes carry no file path. Use `scripts/find_symbol.py` to get `path:line`.
