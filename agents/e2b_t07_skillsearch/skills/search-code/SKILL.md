---
name: search-code
description: Find the functions and classes that match a few words (or a symbol name), with file and line. Use it to locate code when you do not know its file.
---

# search-code

Run the script with two to four words about the code you want, or one symbol name, as one `query` string:

```
run_skill_script(skill_name="search-code", file_path="scripts/search_code.py", args={"query": "upload file size"})
run_skill_script(skill_name="search-code", file_path="scripts/search_code.py", args={"query": "APIRoute.__init__"})
```

It prints the 10 best matches as `path:line  kind  qualified.name` and the definition line. Then read the code
with `read_file`.
