"""Fetch a Kaggle competition write-up as raw markdown (+ its figures), no browser.

The write-up pages are JS-rendered, so a plain GET returns an empty shell. The page
itself loads the text from an internal endpoint that needs only the anonymous
session cookie + XSRF token; this script does the same two requests.

    uv run tools/kaggle_writeup.py <write-up URL> --out comps/<slug>/study/<entry>

Writes <out>/source.md (header + the author's markdown, untouched) and, unless
--no-images, every figure to <out>/figures/NN.<ext>. The endpoint is undocumented:
if it stops answering, fall back to reading the page in Chrome.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import pathlib
import re
import sys
import urllib.request
from datetime import datetime, timezone

UA = "Mozilla/5.0 (X11; Linux x86_64) grandmaster-study"
ENDPOINT = "https://www.kaggle.com/api/i/discussions.WriteUpsService/GetWriteUpBySlug"


def fetch(url: str) -> dict:
    m = re.search(r"kaggle\.com/competitions/([^/]+)/writeups/([^/?#]+)", url)
    if not m:
        sys.exit(f"not a competition write-up URL: {url}")
    comp, slug = m.groups()
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", UA)]
    opener.open(url, timeout=60).read()  # sets the anonymous session + XSRF cookie
    xsrf = next((c.value for c in jar if c.name == "XSRF-TOKEN"), "")
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps({"competitionName": comp, "writeUpSlug": slug, "slug": slug}).encode(),
        headers={"content-type": "application/json", "x-xsrf-token": xsrf},
    )
    body = opener.open(req, timeout=60).read().decode()
    if body.lstrip().startswith("<"):
        sys.exit("endpoint returned HTML, not JSON (changed or blocked) - read the page in Chrome instead")
    return json.loads(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()

    d = fetch(a.url)
    md = d["message"]["rawMarkdown"]
    a.out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    head = (
        f"<!-- raw source, never published. url: {a.url} | fetched: {now} | "
        f"posted: {d['message'].get('postDate', '?')} -->\n"
        f"# {d.get('title', '')}\n{d.get('subtitle', '')}\n\n"
    )
    (a.out / "source.md").write_text(head + md)
    (a.out / "source.json").write_text(json.dumps(d, indent=1))

    imgs = re.findall(r"!\[([^\]]*)\]\(([^)\s]+)", md)
    print(f"source.md: {len(md)} chars, {len(imgs)} figures")
    if a.no_images:
        return
    fig = a.out / "figures"
    fig.mkdir(exist_ok=True)
    for i, (alt, src) in enumerate(imgs, 1):
        if src.startswith("/"):
            src = "https://www.kaggle.com" + src
        try:
            req = urllib.request.Request(src, headers={"User-Agent": UA})
            blob = urllib.request.urlopen(req, timeout=120).read()
        except Exception as e:  # a missing figure must not lose the text
            print(f"  {i:02d} FAILED {src[:80]} ({e})")
            continue
        ext = re.search(r"\.(png|jpe?g|gif|webp|svg)", src.lower())
        p = fig / f"{i:02d}.{ext.group(1) if ext else 'png'}"
        p.write_bytes(blob)
        print(f"  {p.name}  {len(blob) // 1024} KB  {alt[:60]}")


if __name__ == "__main__":
    main()
