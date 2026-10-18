You are testing three skill tools. Do exactly these steps in order, one tool call per turn, and nothing else.

1. Call `load_skill` with `skill_name` set to `find-symbol`.
2. Call `load_skill_resource` with `skill_name` set to `find-symbol` and `file_path` set to `resources/notes.md`.
3. Call `run_skill_script` with `skill_name` set to `find-symbol`, `file_path` set to `scripts/find_symbol.py` and `args` set to the list `["PreparedRequest.copy", "_copy_cookie_jar"]`.
4. Call `run_skill_script` again with the same skill and file, and `args` set to `["no_such_symbol_xyz"]`.
5. Reply with the single word DONE.

Do not edit any file. Do not call any other tool.
