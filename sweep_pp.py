#!/usr/bin/env python3
"""Motivation sweep: grid over (model, seq, gbs, mbs, policy, vp), append to CSV.

For each cell runs ONE torchrun subprocess (one policy), parses per-iter latency
+ peak mem + OOM, writes a CSV row. Resumes by skipping cells already in the CSV.

Grid:
  models: small medium large xl 2.7b
  seq:    1024 2048
  batch:  (mbs,gbs) pairs given below (M=gbs/mbs always multiple of PP)
  policies: gpipe, 1f1b (vp=-, one run) ; interleaved_1f1b, looped_bfs (each valid vp)

Usage:
  nohup python sweep_pp.py \
      --data-path /autopp/data/common_pile/my-gpt3_00_text_document \
      --vocab-file /autopp/data/common_pile/bpe/vocab.json \
      --merge-file /autopp/data/common_pile/bpe/merges.txt \
      --out sweep_results.csv > sweep.log 2>&1 &
"""
import argparse, csv, os, re, statistics, subprocess, sys, itertools

PP = 4
CONFIGS = {
    'small':  dict(layers=12, hidden=768,  heads=12),
    'medium': dict(layers=24, hidden=1024, heads=16),
    'large':  dict(layers=24, hidden=1536, heads=16),
    'xl':     dict(layers=24, hidden=2048, heads=16),  # heads=16 (2048/24 non-integer; keep d_head=128)
    '2.7b':   dict(layers=32, hidden=2560, heads=32),
}
MODELS = ['small', 'medium', 'large', 'xl', '2.7b']
SEQS = [1024, 2048]
# (mbs, gbs) pairs from the user's plan; M=gbs/mbs, all multiples of PP=4
BATCH = []
for gbs in [4, 8, 16, 32, 64]:   BATCH.append((1, gbs))
for gbs in [8, 16, 32, 64]:      BATCH.append((2, gbs))
for gbs in [16, 32, 64]:         BATCH.append((4, gbs))
for gbs in [32, 64]:             BATCH.append((8, gbs))
for gbs in [64]:                 BATCH.append((16, gbs))

NONVP_POLICIES = ['gpipe', '1f1b']
VP_POLICIES = ['interleaved_1f1b', 'looped_bfs']

ITER_RE = re.compile(r'elapsed time per iteration \(ms\):\s*([\d.]+)')
MEM_RE  = re.compile(r'max allocated:\s*([\d.]+)')
OOM_RE  = re.compile(r'out of memory|CUDA out of memory|OutOfMemoryError', re.I)

CSV_FIELDS = ['model','seq','gbs','mbs','M','policy','vp',
              'median_ms','mean_ms','e2e_s','peak_MB','status']

def vp_candidates(layers):
    return [vp for vp in range(2, layers // PP + 1) if layers % (PP * vp) == 0]

def cells():
    """Yield every (model, seq, mbs, gbs, policy, vp) to run."""
    for model in MODELS:
        layers = CONFIGS[model]['layers']
        vps = vp_candidates(layers)
        for seq in SEQS:
            for (mbs, gbs) in BATCH:
                M = gbs // mbs
                if M % PP != 0 or M < PP:
                    continue  # need M multiple of PP and >= PP for a full pipeline
                for pol in NONVP_POLICIES:
                    yield (model, seq, mbs, gbs, M, pol, 0)   # vp=0 means N/A
                for pol in VP_POLICIES:
                    for vp in vps:
                        yield (model, seq, mbs, gbs, M, pol, vp)

def build_cmd(model, seq, mbs, gbs, pol, vp, a):
    c = CONFIGS[model]
    env = dict(os.environ)
    env.update(NCCL_P2P_DISABLE='1', NCCL_SHM_DISABLE='1', PIPELINE_SCHEDULE=pol)
    cmd = [
        'torchrun', f'--nproc_per_node={PP}', a.script,
        '--num-layers', str(c['layers']),
        '--hidden-size', str(c['hidden']),
        '--num-attention-heads', str(c['heads']),
        '--seq-length', str(seq), '--max-position-embeddings', str(seq),
        '--pipeline-model-parallel-size', str(PP), '--tensor-model-parallel-size', '1',
        '--micro-batch-size', str(mbs), '--global-batch-size', str(gbs),
        '--train-iters', str(a.warmup + a.measure), '--log-interval', '1',
        '--eval-interval', '1000000', '--eval-iters', '0', '--split', '100,0,0',
        '--lr', '1e-4',
        '--data-path', a.data_path, '--vocab-file', a.vocab_file,
        '--merge-file', a.merge_file, '--tokenizer-type', 'GPT2BPETokenizer',
        '--no-gradient-accumulation-fusion', '--no-masked-softmax-fusion', '--bf16',
    ]
    if pol in VP_POLICIES:
        cmd += ['--num-layers-per-virtual-pipeline-stage', str(c['layers'] // (PP * vp))]
    return cmd, env

def run_cell(model, seq, mbs, gbs, pol, vp, a):
    cmd, env = build_cmd(model, seq, mbs, gbs, pol, vp, a)
    iters, peak, oom, crashed, tail = [], 0.0, False, False, []
    try:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            tail.append(line);  tail = tail[-30:]
            if OOM_RE.search(line): oom = True
            m = ITER_RE.search(line)
            if m: iters.append(float(m.group(1)))
            m = MEM_RE.search(line)
            if m: peak = max(peak, float(m.group(1)))
        rc = p.wait()
        if rc != 0 and not iters and not oom:
            crashed = True
    except Exception:
        crashed = True
    steady = iters[a.warmup:] if len(iters) > a.warmup else []
    if oom:
        status, med, mean, e2e = 'OOM', '', '', ''
    elif crashed or not steady:
        status, med, mean, e2e = 'ERROR', '', '', ''
        if crashed: print(''.join(tail[-10:]), flush=True)
    else:
        status = 'OK'
        med = round(statistics.median(steady), 1)
        mean = round(statistics.mean(steady), 1)
        e2e = round(sum(iters) / 1000, 1)
    return dict(model=model, seq=seq, gbs=gbs, mbs=mbs, M=gbs//mbs, policy=pol,
                vp=(vp if pol in VP_POLICIES else ''),
                median_ms=med, mean_ms=mean, e2e_s=e2e,
                peak_MB=(round(peak) if peak else ''), status=status)

def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add((row['model'], row['seq'], row['gbs'], row['mbs'],
                          row['policy'], row['vp']))
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--vocab-file', required=True)
    ap.add_argument('--merge-file', required=True)
    ap.add_argument('--script', default='pretrain_gpt.py')
    ap.add_argument('--out', default='sweep_results.csv')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--measure', type=int, default=80)
    a = ap.parse_args()

    all_cells = list(cells())
    done = load_done(a.out)
    new_file = not os.path.exists(a.out)
    f = open(a.out, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new_file:
        w.writeheader(); f.flush()

    total = len(all_cells)
    print(f"total cells: {total} | already done: {len(done)}", flush=True)
    for i, (model, seq, mbs, gbs, M, pol, vp) in enumerate(all_cells, 1):
        key = (model, str(seq), str(gbs), str(mbs), pol,
               str(vp if pol in VP_POLICIES else ''))
        if key in done:
            continue
        tag = f"{model} seq{seq} gbs{gbs} mbs{mbs} M{M} {pol}" + (f" vp{vp}" if pol in VP_POLICIES else "")
        print(f"\n[{i}/{total}] {tag}", flush=True)
        row = run_cell(model, seq, mbs, gbs, pol, vp, a)
        w.writerow(row); f.flush()
        print(f"   -> {row['status']} median={row['median_ms']} peak={row['peak_MB']}", flush=True)
    f.close()
    print("\nSWEEP COMPLETE", flush=True)

if __name__ == '__main__':
    main()
