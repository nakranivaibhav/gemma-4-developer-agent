"""Read the Claude subscription's live 5-hour / 7-day utilization, the way ~/seats on revideo does (2026-10-24).

One 1-token request with the local Claude Code login; the unified rate-limit headers on the response (429 included)
carry the utilization fractions and reset times. The token is read fresh from ~/.claude/.credentials.json on every
probe (Claude Code refreshes it) and never printed or stored.

    envs/judge/bin/python training/judge/usage_guard.py      # prints {"h5": 0.72, "h5_reset": ..., "d7": ...}
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.anthropic.com/v1/messages"
PROBE = {"model": "claude-haiku-4-5-20251001", "max_tokens": 1,
         "messages": [{"role": "user", "content": "hi"}],
         "system": "You are Claude Code, Anthropic's official CLI for Claude."}
CREDS = Path.home() / ".claude/.credentials.json"


def probe():
    tok = json.loads(CREDS.read_text())["claudeAiOauth"]["accessToken"]
    req = urllib.request.Request(API, data=json.dumps(PROBE).encode(), headers={
        "authorization": f"Bearer {tok}", "anthropic-beta": "oauth-2025-04-20",
        "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            hdr, code = r.headers, r.status
    except urllib.error.HTTPError as e:
        hdr, code = e.headers, e.code
    except Exception as e:  # network trouble: no reading
        return {"error": str(e)[:120], "at": time.time()}

    def num(k):
        try:
            return float(hdr.get(f"anthropic-ratelimit-unified-{k}"))
        except (TypeError, ValueError):
            return None

    return {"http": code, "status": hdr.get("anthropic-ratelimit-unified-status"),
            "h5": num("5h-utilization"), "h5_reset": num("5h-reset"),
            "d7": num("7d-utilization"), "d7_reset": num("7d-reset"), "at": time.time()}


if __name__ == "__main__":
    print(json.dumps(probe()))


CAP_FILE = Path("/home/vaibhav/data/gemma4/judge/usage_cap.json")  # {"cap_5h": 0.85, "cap_7d": 1.0}; read live


class Cap:
    """Async gate for runners: `await cap.wait()` before starting a task; blocks while 5h or 7d utilization is at or
    above its cap, until that window resets (heartbeat line every 10 min). One probe per `every` seconds is shared.
    If CAP_FILE exists, its caps override the constructor's on every probe, so a running job can be re-capped live."""

    def __init__(self, cap_5h=0.90, cap_7d=1.0, every=60.0, log=print):
        import asyncio
        self.caps, self.every, self.log = (("h5", cap_5h), ("d7", cap_7d)), every, log
        self.lock, self.at, self.r = asyncio.Lock(), 0.0, None

    async def wait(self):
        import asyncio
        async with self.lock:
            while True:
                if time.time() - self.at > self.every:
                    self.r, self.at = await asyncio.to_thread(probe), time.time()
                    try:
                        c = json.loads(CAP_FILE.read_text())
                        self.caps = (("h5", float(c.get("cap_5h", 1.0))), ("d7", float(c.get("cap_7d", 1.0))))
                    except (OSError, ValueError):
                        pass
                r = self.r or {}
                over = [(w, r.get(w), r.get(w + "_reset")) for w, c in self.caps if r.get(w) is not None and r[w] >= c]
                if not over:
                    return
                w, u, reset = max(over, key=lambda x: x[2] or 0)
                left = max(60.0, (reset or time.time() + 600) - time.time() + 60)
                self.log(f"CAP {w} {u:.0%} >= cap -> no new tasks for {left / 60:.0f} min")
                while left > 0:
                    await asyncio.sleep(min(600, left))
                    left -= 600
                    if left > 0:
                        self.log(f"CAP waiting, {left / 60:.0f} min left")
                self.at = 0.0
