"""STEP 4: compare runtime _TRACE dumps against the static paper schedule.

Usage:
  python compare_trace.py traces/gpipe            --policy gpipe            --pp 4 --vp 1 --m 8
  python compare_trace.py traces/1f1b             --policy 1f1b             --pp 4 --vp 1 --m 8
  python compare_trace.py traces/interleaved_1f1b --policy interleaved_1f1b --pp 4 --vp 2 --m 8
  python compare_trace.py traces/looped_bfs       --policy looped_bfs       --pp 4 --vp 2 --m 8

Reads trace_rank{r}.json (each a list of [rank, 'F'|'B', mb, chunk]) and checks,
per rank, that the (kind, mb, chunk) sequence equals the static order the engine
*should* produce for that policy. Prints PASS/FAIL + first divergence.
"""
import argparse, glob, json, os, sys

# ---- static schedule model (same logic as verify_schedules.py) ----
def schedule_table(M, VP, g):
    tbl = []
    for start in range(0, M, g):
        rng = range(start, M) if start + g >= M else range(start, start + g)
        for chunk in range(VP):
            for mb in rng:
                tbl.append((mb, chunk))
    return tbl

def static_order(policy, PP, VP, M, r):
    if VP == 1:  # gpipe / 1f1b
        W = M if policy == 'gpipe' else min(PP - r - 1, M)
        fwd = [('F', mb, 0) for mb in range(M)]
        bwd = [('B', mb, 0) for mb in range(M)]
        order = fwd[:W]
        for i in range(M - W):
            order += [fwd[W + i], bwd[i]]
        order += bwd[M - W:]
        return order
    # interleaved engines
    total = M * VP
    g = M if policy == 'looped_bfs' else PP
    tbl = schedule_table(M, VP, g)
    W = total if policy == 'looped_bfs' else min((PP - r - 1) * 2 + (VP - 1) * g, total)
    fwd = [('F', mb, ch) for (mb, ch) in tbl]
    bwd = [('B', mb, VP - 1 - ch) for (mb, ch) in tbl]
    order = fwd[:W]
    for k in range(W, total):
        order += [fwd[k], bwd[k - W]]
    order += bwd[total - W:]
    return order

def norm(ev):  # trace event [rank,kind,mb,chunk] -> (kind,mb,chunk)
    return (ev[1], int(ev[2]), int(ev[3]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace_dir')
    ap.add_argument('--policy', required=True,
                    choices=['gpipe', '1f1b', 'interleaved_1f1b', 'looped_bfs'])
    ap.add_argument('--pp', type=int, required=True)
    ap.add_argument('--vp', type=int, required=True)
    ap.add_argument('--m', type=int, required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.trace_dir, 'trace_rank*.json')))
    if not files:
        print(f"no trace_rank*.json in {a.trace_dir}"); sys.exit(1)

    all_ok = True
    for f in files:
        r = int(os.path.basename(f).split('rank')[1].split('.')[0])
        actual = [norm(e) for e in json.load(open(f))]
        expect = static_order(a.policy, a.pp, a.vp, a.m, r)
        if actual == expect:
            print(f" rank{r}: PASS  ({len(actual)} ops)")
        else:
            all_ok = False
            # find first divergence
            i = next((j for j in range(min(len(actual), len(expect)))
                      if actual[j] != expect[j]), min(len(actual), len(expect)))
            print(f" rank{r}: FAIL  (len actual={len(actual)} expect={len(expect)})")
            print(f"        first diff at step {i}: actual={actual[i] if i<len(actual) else '-'} "
                  f"expect={expect[i] if i<len(expect) else '-'}")
            lo, hi = max(0, i - 2), i + 3
            print(f"        actual[{lo}:{hi}] = {actual[lo:hi]}")
            print(f"        expect[{lo}:{hi}] = {expect[lo:hi]}")
    print(f"\n=== {a.policy}: {'ALL RANKS PASS — matches paper schedule' if all_ok else 'MISMATCH'} ===")
    sys.exit(0 if all_ok else 2)

if __name__ == '__main__':
    main()
