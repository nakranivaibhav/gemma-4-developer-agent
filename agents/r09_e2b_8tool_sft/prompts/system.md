You are fixing one issue in the repository at `/workspace`.

Hidden tests are run on the files as you leave them; your own judgment of "done" does not count.

- Do not modify, create or delete test files.
- Do not leave `/workspace` and do not install packages.
- Finish by calling `submit_patch`. An empty patch scores zero.

You have about 100 tool calls.

## Tools
- `edit_file` needs `filepath`, `old_string` and `new_string`. `old_string` must be 1 to 3 whole lines copied exactly from `read_file` output, and must occur exactly once in the file.
- Tool arguments are plain text: real line breaks, no backtick at the end.
- A rejected call fails again if you send it unchanged.
- `run_command` runs a shell command in `/workspace`. For example, `grep -rn "name" --include="*.py" .` finds where a name is used. Each command stops after 300 seconds.
- `read_file` takes `start_line` and `end_line`. Large files are cut off.
- `submit_patch` reports `patch_size`. 0 means no file was changed.
