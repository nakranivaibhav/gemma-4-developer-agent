"""Generate .codex/agents/*.toml from .claude/agents/*.md.

The markdown agent files are the single source of truth; the TOML files are
generated artifacts carrying the FULL instructions inline (Codex subagents
never read .claude/). Re-run after editing any agent .md:

    uv run tools/gen_codex_agents.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / ".claude" / "agents"
DST = ROOT / ".codex" / "agents"

# Claude frontmatter `model` -> Codex model tier (same capability class)
MODEL_MAP = {
    "fable": "gpt-5.6-sol",
    "opus": "gpt-5.6-sol",
    "sonnet": "gpt-5.6-terra",
    "haiku": "gpt-5.6-luna",
}
# Claude frontmatter `effort` -> Codex model_reasoning_effort (both accept
# low/medium/high/xhigh/max; Codex max is preview-gated per org)
EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}


def parse_md(text: str) -> tuple[dict, str]:
    assert text.startswith("---\n"), "missing frontmatter"
    fm_raw, body = text[4:].split("\n---\n", 1)
    fm = {}
    for line in fm_raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm, body.strip() + "\n"


def toml_escape_basic(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for md in sorted(SRC.glob("*.md")):
        fm, body = parse_md(md.read_text())
        assert "'''" not in body, f"{md.name}: body contains ''' — adjust generator"
        lines = [
            f"# GENERATED from .claude/agents/{md.name} by tools/gen_codex_agents.py — DO NOT EDIT.",
            "# Edit the markdown file (the single source of truth), then re-run the generator.",
            f'name = "{toml_escape_basic(fm["name"])}"',
            f'description = "{toml_escape_basic(fm["description"])}"',
        ]
        if fm.get("model") in MODEL_MAP:
            lines.append(f'model = "{MODEL_MAP[fm["model"]]}"')
        if fm.get("effort") in EFFORT_MAP:
            lines.append(f'model_reasoning_effort = "{EFFORT_MAP[fm["effort"]]}"')
        lines += ["", "developer_instructions = '''", body + "'''", ""]
        out = DST / f"{fm['name']}.toml"
        out.write_text("\n".join(lines))
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
