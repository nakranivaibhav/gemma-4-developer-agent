---
name: find-symbol
description: Find the file and line where a class, method or function is defined in the repository. Use it before reading or editing code when you know a symbol name but not its file.
---

# find-symbol

Run the script with one or more symbol names. It prints `path:line  kind  qualified.name` for every definition whose name matches.

```
run_skill_script(skill_name="find-symbol", file_path="scripts/find_symbol.py", args=["PreparedRequest", "copy"])
```

A dotted name such as `Session.send` matches only that method. A bare name matches every definition with that name.
Read `resources/notes.md` for what the code graph tools do not cover.
