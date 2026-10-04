---
name: kaggle-dashboard
description: Build ONE private artifact page that lets a human see, at a glance, what the current model is and where its errors live, so they can review it and steer the search. Use when the user asks for a dashboard / a visual / "show me the model" / "where are the errors", or after a promotion. Instructions only, no fixed template - read the comp, build the page, publish it, return the link.
argument-hint: "[slug] [node or run to show]  (omit = the only comp, its champion / served chain)"
---

# /kaggle-dashboard - one page: what the model is, where its errors live

**Reader:** a smart human who is NOT inside this search. They will look for two
minutes and then tell the agent what to try. Every choice below serves that:
few things on screen, plain words, real numbers, worst cases first.

**Output:** one private artifact link. Nothing else counts as done.

## Limits (non-negotiable)
- **Read-only on the comp.** Write only under `comps/<slug>/dashboard/`. Never
  append to `journal.md`, never edit a node, never train, never submit.
- `uv run` for everything; timestamps from `date -u +%Y-%m-%dT%H:%MZ`.
- One heavy process at a time: check `nvidia-smi` and `free -g` first. A single
  inference forward pass on ONE example is allowed; anything longer is not.
- Filter and truncate every read of `journal.md` and logs (`tools/jgrep.sh`).
- The words ceiling / exhausted / impossible / nothing left / practical limit
  never appear on the page.
- **Time-box, then degrade.** ~30 min per panel. A panel you cannot fill ships
  as a plain labelled box: "not captured: <one-line reason>". A missing panel
  is fine; a fix-loop is not.

## 1 - Find the facts (read, don't recompute)
1. `spec.md` -> the official metric and its terms. `state.md` -> which node /
   run is the champion or the served chain. Show THAT one unless told otherwise.
2. The **per-unit results table** for that run (one row per clip / fold unit /
   image, with the metric's counts). Prefer the table the official metric code
   wrote. Recompute only if none exists.
3. The model code: the inference entry script and the module it builds.
4. `LEVER_LEDGER.md` / `docs/*METRIC*` if present, for how errors are bucketed.

## 2 - The page answers three questions, in this order
**Q1 "What is the model?"** A left-to-right strip of **4-8 stages** - the whole
pipeline from raw input to submission, not just the network, and never a
layer-by-layer graph. Each box: plain name, shape in -> out, parameter count
(or "no parameters"), and **the errors charged to this stage**. The strip IS
the error budget: the human should see which box leaks most. Click a box ->
what flows through it on one real example (section 3).

**Q2 "Where does the score go?"** One bar from the current score to a perfect
one, split by error bucket, in score points and in counts. Under it a **unit
grid**: one tile per unit, coloured by score, sized by metric weight, sorted
worst first; if units come from different sources, group them. State how
concentrated the errors are ("9 of 39 units hold 60%").

**Q3 "Show me the worst."** Click a tile -> that unit's own numbers and a view
of its actual mistakes. Worst ~8 units only. 3D data gets a 3D view (rotate /
zoom / pan) with a time slider if it has time; image data gets the image with
errors drawn on; tabular gets the worst rows. Errors coloured by bucket, with
the same colours as Q2.

Rules for the numbers:
- **Every error is charged to exactly one bucket and one stage, and the buckets
  add up to the total.** Assert it in the build script. If the metric gives no
  clean partition, say so on the page instead of inventing one.
- **Every panel prints its population:** which run, which unit pool, how many
  units, when built. Numbers from two pools never share a panel.
- Low-count terms show their counts (a Jaccard built on 25 events must say 25).
- If a number you compute disagrees with `state.md` or the journal, do not pick
  one - show yours, and report the disagreement in your final message.

## 3 - Forward trace (the "click a stage" content)
One real example, `model.eval()`, `torch.no_grad()`. `register_forward_hook` on
the **named top-level blocks** (the stages), not every leaf. Per block record:
output shape, mean, std, fraction of exact zeros. For spatial outputs save up
to 4 channels (highest variance) as a mid-slice or max-projection thumbnail,
<= 96 px, base64 PNG. Remove the hooks. Non-network stages (thresholds,
matching, solvers) show counts in -> out instead (e.g. candidates -> kept).
Not a torch model? The strip still stands; the trace becomes feature
importance or stage in/out counts.

## 4 - How to build it (this is what prevents the fix-loop)
- **One build script** `comps/<slug>/dashboard/build.py` reads the sources,
  runs the asserts, and writes `index.html` by injecting **one JSON blob** into
  the page. No number is ever typed into HTML by hand.
- JSON safety: cast numpy types, turn NaN/inf into `null`
  (`json.dumps(..., allow_nan=False)`), round floats to 4 significant digits,
  and replace `</` with `<\/` before placing it in
  `<script type="application/json" id="data">`.
- Size: aim under 3 MB (hard limit 16 MB). Worst ~8 units only, error
  neighbourhoods not whole volumes, <= 40 thumbnails.
- Scripts only from `cdnjs.cloudflare.com` or `cdn.jsdelivr.net/npm/`, version
  pinned; run `curl -sI <url>` once to confirm 200 before using it. Fonts only
  from Google Fonts. Everything else inline. No `fetch()` of local files.
- Libraries first: Plotly (`scatter3d`) already gives rotate / zoom / pan. Set
  a constant `uirevision` so the camera does NOT reset when the time slider
  moves; plot in physical units with `aspectmode: 'data'`; update with
  `Plotly.react`, never re-create the plot.
- Wrap each panel's render in its own `try/catch` that prints the error inside
  that panel - one bug must not blank the page.
- The page opens **already showing something**: worst unit selected, leakiest
  stage selected. No empty shell waiting for a click.
- Colour tokens on `:root`; dark mode under
  `@media (prefers-color-scheme: dark)` guarded by
  `:root:not([data-theme="light"])`, and again under
  `:root[data-theme="dark"]`; explicit `body` background; chart text and
  backgrounds read from the same tokens. Works at phone width: 16 px gutter,
  wide things scroll inside their own box, never the page.
- One accent colour plus one colour per error bucket, reused everywhere. Plain
  words on every label ("extra links", not "edge_fp"); the raw column name may
  follow in small type.
- `<title>`: a 2-4 word name. The explanation goes in the publish `description`.

## 5 - Publish and return
1. First call `Artifact` with `action: "quickstart"`, `intent: "other"`; if
   its result carries no page-design guidance, load the `artifact-design` skill.
2. **Look once:** one headless screenshot of the local file (plus its console
   errors, if that comes free), one pass of edits, no second look.
3. If `comps/<slug>/dashboard/URL.txt` exists: `Artifact` `action: "read"` that
   URL, then publish with `url` so the link stays the same. Otherwise publish
   new (`icon: "chart"`) and write the URL to that file.
4. Final message: the link - one line per panel on what it shows - anything
   degraded and why - any number that disagreed with the comp's own records.
   The main session tells the human and journals it; you do not.
