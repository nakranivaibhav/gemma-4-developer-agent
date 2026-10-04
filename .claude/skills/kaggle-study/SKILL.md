---
name: kaggle-study
description: Learn a top solution together with the human. Build ONE private artifact page of intuitive, mostly-3D scenes (one per pipeline stage), then walk the stages in chat one message at a time so the page and the text go hand in hand; close each entry by agreeing with the human what "we could have done" and logging only that for future comps. Use when the human wants to understand a top submission / write-up / winning notebook - "explain the Nth place solution", "walk me through this write-up", "study the top subs", "/kaggle-study".
argument-hint: "[slug] <write-up URL | rank | notebook or repo path>  (one entry per run)"
---

# /kaggle-study - learn one top solution: a 3D page + a chat walk + an agreed log

**Reader:** the human, learning. Not a report for a team. Done = they can say
what each stage takes in, does and gives out, and the log holds the lessons
THEY agreed to.

**Three outputs, in this order:** (1) one private artifact link, (2) the walk
in chat, one stage per message, (3) the agreed "could have done" entries.
Runs in the **main session** - you must know every scene to talk about it and
to patch it while the human is looking at it. No subagent.

## Limits
- One entry (one write-up / notebook / repo) per run. Write only under
  `comps/<slug>/study/<entry>/` (entry = `03-teamname`), plus at the close:
  the log file and ONE journal line.
- Never train, never submit. `uv run`; dates from `date -u`; filter + truncate
  every journal read (`tools/jgrep.sh`); the banned words stay off the page
  and out of the log.
- **Nothing is logged that the human did not agree to, in their wording.**
- Time-box, then degrade: a scene that fights you ships as a labelled box
  ("not built yet: <reason>") and gets fixed before its stage comes up in the
  walk. No fix-loops.

## 1 - Read the source (facts before pictures)
1. Save the raw text FIRST to `source.md` with the URL and the UTC time; work
   from the file. Kaggle write-ups:
   `uv run tools/kaggle_writeup.py <url> --out comps/<slug>/study/<entry>`
   (no browser; writes `source.md` + `figures/`). READ the architecture
   figures - they carry shapes and details the prose leaves out. Only if the
   tool fails: Chrome page-text, or ask the human to paste. If the team
   published code, pull it: code beats prose wherever the prose is vague.
2. Write `stages.md`: the pipeline as 4-8 stages. Per stage: why it exists -
   what goes in (shape, units) - the model / algorithm - what comes out - how
   it is trained - what it bought (their number).
3. Tag every fact **says** (in the source) / **inferred** (yours) / **not
   stated**. Never fill a silence with a plausible default. Before calling a
   setting "their choice", check whether it is just the library's default.
4. Their numbers are their CV on their folds: write which (CV / public /
   private) beside each one. A row that adds several things at once cannot be
   split between them - say so.
5. **Vs ours**, per stage: read it from our code / journal NOW and cite
   `file:line`. Never from memory (biohub: a separate helper detector got
   described as a "head" of the main model). Artifacts gone -> write "not on
   disk"; do not reconstruct.
6. From the second entry of a comp on: mark each stage "same idea as <entry>"
   or "new". A thing several top teams did independently is the strongest
   lesson there is.

## 2 - The page: one scene per stage
Every scene must:
- **Show the operation happening to data** - the input, then the step, then
  the output - behind a Step 1 / 2 / 3 control. The chat uses the same step
  numbers.
- **Show the failure the stage fixes:** a with / without toggle (flow on/off,
  ignore-zones on/off), so the "why" is seen, not asserted.
- **Carry a theirs / ours toggle** on the same example wherever we had that
  stage.
- **Use one running example** through all scenes: the same small
  neighbourhood (10-30 objects) and one "hero" to follow. The whole volume is
  a toggle, never the default - a cloud of 300 points teaches nothing.
- **Put the tensor's-eye view beside the 3D:** a small 2D inset of what the
  model actually receives or emits (a slice, a target, a row of scores), and a
  strip of boxes with the shapes in -> out.
- **Draw a network as its feature maps** (they are 3D objects): slabs to
  scale, labelled with shape and channels, pretrained parts marked, widths the
  source does not give shown as "not stated".
- **Be true to scale:** physical units, real proportions, real radii and
  thresholds, real label rates. Any exaggeration is printed on the scene
  ("arrows 8x").
- One colour = one thing on every scene; labels sit ON the objects; a
  one-sentence caption says what to notice. Cannot write that sentence -> cut
  the scene.

3D where the thing is 3D: volumes, point clouds, space-time, feature maps, an
embedding or a decision surface over three features. Tables and score ladders
stay 2D. **Real data first:** one real unit from `data/` or from artifacts we
kept; otherwise simulate and print "simulated" on the scene.

Page furniture: a sticky stage bar with the SAME numbers and names the chat
uses - scene ids ("3.1") - 1-3 "try this" lines under each scene - an opening
map (the whole pipeline as a strip with shapes and the score after each
stage) - a glossary and a "your questions" list per stage, both filled during
the walk.

## 3 - How to build it (this is what prevents the fix-loop)
- Multi-file page under `study/<entry>/page/`: `index.html` (text, stage bar,
  containers), `runtime.js` (shared: renderer, orbit controls, colour key,
  stepper, labels), `scenes/<id>.js` (one small file per scene), `data/*.js`.
  Patching a scene = editing one small file. An earlier study page exists?
  Start from its `runtime.js`.
- Scene and data files are **classic scripts that register themselves**
  (`STUDY.scene("3.1", build)`; data as `window.DATA_x = ...`). No ES modules,
  no `fetch()` of local files. Publish `index.html` as the page and every
  other file through `files`, same relative paths.
- three.js r128 UMD from cdnjs + `OrbitControls` from
  `cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/`. `curl -sI` each
  URL once before using it.
- **One WebGL renderer for the whole page** (or create on enter / dispose on
  leave). Browsers drop the oldest context past ~8-16 - it shows up as scenes
  silently going blank.
- Instanced meshes; <= ~5k objects per scene; render only the scenes in view;
  respect reduced motion; orbit + zoom + pan + a reset-view button;
  auto-rotate stops on first touch. Page under ~3 MB.
- Each scene builds inside its own `try/catch` that prints the error in its
  own box - one bug must not blank the page.
- Keep the viewer's place across a republish (stage, step, camera) - see
  "Open viewers" in `artifact-design`; `localStorage` only inside try/catch.
- Load `artifact-design` first (`dataviz` for any chart). Both themes; the 3D
  panels keep one dark ground. Our own words plus a link to the source;
  `source.md` is never published.
- **Look once:** publish first (`icon: "cube"`), write the link to `URL.txt`,
  then open the artifact link in Chrome, scroll through the scenes one time,
  one pass of fixes, republish. (Chrome runs on another machine and cannot
  reach localhost here.)
  Every later publish uses the same path, so the link never changes.

## 4 - The walk (chat and page hand in hand)
- Open with: the link, the map in five lines, and "Stage 1?".
- **One stage per message.** First line says where to look: "Page: Stage 3,
  scene 3.1, press Step 2". Then: why the stage exists - in / out - how it
  works, in numbered steps that match the scene's steps - how it is trained -
  what it bought - vs ours (verified) - what the source does not say. Plain
  words, every term explained the first time, and talk about what they SEE
  ("the three bright planes are the three input channels").
- Answer these before they are asked, per stage: what the model actually is
  (backbone, 2D or 3D, pretrained or not), which choices are library
  defaults, and what our equivalent was.
- End with a check: anything fuzzy, or go on? Never run two stages ahead.
- **The page follows the conversation.** When a question shows that a scene
  does not carry the idea, or a term needs a picture: patch that scene or add
  a small one, add the question to that stage's "your questions" list,
  republish, and say what changed. Open pages update themselves.
- While walking, append each real difference to `gaps.md`: what they did -
  what we did (`file:line`) - what it was worth to them. Keep a
  `walked: 1,2,3` line at its top. After a compaction or in a new session,
  resume from `stages.md` + `gaps.md` + `URL.txt`.

## 5 - Close the entry: "what we could have done"
1. Lay out the candidates from `gaps.md`, one row each: what they did - what
   we did - what it bought THEM (their ablation; never written as "our gain")
   - could we have seen it (the earliest evidence in our journal, or the idea
   we had dropped, with its date) - what it would have cost us - the lesson
   as "when <situation>, <action>".
2. **Discuss it with the human. They pick what goes in and how it is worded.**
   Hindsight is labelled as hindsight.
3. Then, artifact-then-mark:
   - append the agreed entries to `~/.grandmaster/top_solutions.md` (private,
     outside every repo, the same path from every comp; create on first use):
     ```
     ## <slug> - <rank> place (<team>) - studied <UTC date>
     task: <data type, metric, what made it hard>
     source: <url> - page: <link> - theirs: <score, which LB> - ours: <score>
     - **When <situation>, <action>.** They: ... We: ... Seen-it: ...
     ```
   - add the same list as the last section of the page and republish;
   - append ONE `OUTSIDE` line to `journal.md` naming `study/<entry>/stages.md`
     and the page link, then re-render.
4. First entry studied after a comp has closed: also list our own case-bank
   lines (`MEMORY.md`) that the final result contradicted, and ask whether to
   amend them. A wrong lesson travels too.

Final message of a run: the link - the agreed entries as logged - anything on
the page that shipped degraded.
