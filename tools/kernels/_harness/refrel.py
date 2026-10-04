"""Reference-relative parity verdict (README "Amendments"): candidate vs incumbent vs fp64 oracle.

Reads three parity.json files written by parity.py over the SAME --shapes list and seed. A cell is one
(shape x tensor). Verdict: absolute gate passes if the candidate passes the table on every cell; else, if the
oracle fails anywhere (absolute gate unmeetable), pass iff candidate max_abs <= incumbent max_abs on EVERY cell
and the candidate is deterministic. Prints one line per cell that is worse than the incumbent and a final
REFREL_RESULT line.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", default=None, help="optional json path for the verdict")
    ap.add_argument("--tie-abs", type=float, default=0.0,
                    help="cand may exceed the incumbent by at most this absolute amount and count as a tie; only for a "
                         "MEASURED run-to-run spread of the fp64 reference itself (e.g. atomics in its backward). "
                         "Default 0 = strict.")
    a = ap.parse_args(argv)
    c, r, o = (json.load(open(p))["results"] for p in (a.cand, a.incumbent, a.oracle))
    assert len(c) == len(r) == len(o), "the three runs must use the same shape list"
    cells, worse, ratios = 0, [], []
    for rc, rr, ro in zip(c, r, o):
        assert rc["shape_spec"] == rr["shape_spec"] == ro["shape_spec"], "shape lists differ"
        for tc, tr in zip(rc["tensors"], rr["tensors"]):
            cells += 1
            ratios.append(tc["max_abs"] / tr["max_abs"] if tr["max_abs"] > 0 else (0.0 if tc["max_abs"] == 0 else float("inf")))
            if tc["max_abs"] > tr["max_abs"] + a.tie_abs:
                worse.append((rc["shape_spec"], tc["name"], tc["max_abs"], tr["max_abs"]))
    abs_pass = all(x["passed"] for x in c)
    oracle_fails = not all(x["passed"] for x in o)
    det = all(x["deterministic"] for x in c)
    rel_pass = det and not worse
    passed = abs_pass or (oracle_fails and rel_pass)
    for s, n, cm, im in worse:
        print(f"WORSE shape={json.dumps(s)} {n} cand={cm:.3e} incumbent={im:.3e}")
    rec = dict(cells=cells, abs_pass=abs_pass, oracle_fails=oracle_fails, deterministic=det,
               n_worse=len(worse), worst_ratio=max(ratios), passed=passed,
               worse=[dict(shape=s, tensor=n, cand=cm, incumbent=im) for s, n, cm, im in worse])
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    print(f"REFREL_RESULT passed={passed} cells={cells} n_worse={len(worse)} worst_ratio={max(ratios):.3f} "
          f"abs_pass={abs_pass} oracle_fails={oracle_fails} deterministic={det}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
