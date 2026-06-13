#!/usr/bin/env python3
"""STEP 1 measurement harness: run 4 PP schedules, parse per-iteration latency
+ peak memory + OOM, print a comparison table.

Runs each policy as a separate torchrun subprocess (one schedule per process,
since the BFS branch mutates config in-place). Parses Megatron stdout for:
  - 'elapsed time per iteration (ms): X'   -> per-iter latency
  - 'max allocated: X'                      -> peak memory (max over ranks)
OOM / crash -> recorded as OOM for that policy.

Usage:
  python measure_pp.py --config small --gbs 8 --mbs 1 --seq 1024 \
      --warmup 50 --measure 100
(edit BASE_ARGS / CONFIGS / launch command for your tree)
"""
import argparse, re, statistics, subprocess, sys, os

# GPT-3 ladder (n_layers, hidden, heads). seq/batch are swept via CLI.
CONFIGS = {
    'small':  dict(layers=12, hidden=768,  heads=12),
    'medium': dict(layers=24, hidden=1024, heads=16),
    'large':  dict(layers=24, hidden=1536, heads=16),
    'xl':     dict(layers=24, hidden=2048, heads=16),
    '2.7b':   dict(layers=32, hidden=2560, heads=32),
}

# VP on/off per policy. PP fixed = 4.
POLICIES = {
    'gpipe':            dict(vp=False),
    '1f1b':             dict(vp=False),
    'interleaved_1f1b': dict(vp=True),
    'looped_bfs':       dict(vp=True),
}

ITER_RE = re.compile(r'elapsed time per iteration \(ms\):\s*([\d.]+)')
MEM_RE  = re.compile(r'max allocated:\s*([\d.]+)')
OOM_RE  = re.compile(r'out of memory|CUDA out of memory|OutOfMemoryError', re.I)

def build_cmd(policy, cfg, a):
    c = CONFIGS[cfg]
    env = dict(os.environ)
    env.update(NCCL_P2P_DISABLE='1', NCCL_SHM_DISABLE='1', PIPELINE_SCHEDULE=policy)
    cmd = [
        'torchrun', f'--nproc_per_node={a.gpus}', a.script,
        '--num-layers', str(c['layers']),
        '--hidden-size', str(c['hidden']),
        '--num-attention-heads', str(c['heads']),
        '--seq-length', str(a.seq),
        '--max-position-embeddings', str(a.seq),
        '--pipeline-model-parallel-size', str(a.pp),
        '--tensor-model-parallel-size', '1',
        '--micro-batch-size', str(a.mbs),
        '--global-batch-size', str(a.gbs),
        '--train-iters', str(a.warmup + a.measure),
        '--log-interval', '1',
        '--eval-interval', '1000000', '--eval-iters', '0', '--split', '100,0,0',
        '--lr', '1e-4',
        '--data-path', a.data_path,
        '--vocab-file', a.vocab_file, '--merge-file', a.merge_file,
        '--tokenizer-type', 'GPT2BPETokenizer',
        '--no-gradient-accumulation-fusion', '--no-masked-softmax-fusion', '--bf16',
    ]
    if POLICIES[policy]['vp']:
        denom = a.pp * a.vp
        if c['layers'] % denom != 0:
            raise SystemExit(
                f"[{policy}] num_layers={c['layers']} not divisible by PP*VP={denom}; "
                f"pick --vp so that layers % (PP*VP) == 0")
        layers_per_vstage = c['layers'] // denom
        cmd += ['--num-layers-per-virtual-pipeline-stage', str(layers_per_vstage)]
    return cmd, env

def run_one(policy, a):
    cmd, env = build_cmd(policy, CONFIGS_KEY, a) if False else build_cmd(policy, a.config, a)
    iters, peak_mem, oom, crashed = [], 0.0, False, False
    tail = []
    try:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            tail.append(line)
            if len(tail) > 30:
                tail.pop(0)
            if OOM_RE.search(line):
                oom = True
            m = ITER_RE.search(line)
            if m:
                iters.append(float(m.group(1)))
            m = MEM_RE.search(line)
            if m:
                peak_mem = max(peak_mem, float(m.group(1)))
        rc = p.wait()
        if rc != 0 and not iters and not oom:
            crashed = True   # non-OOM failure (config error, assert, etc.)
    except Exception:
        crashed = True
    if crashed:
        # surface the real reason instead of mislabeling as OOM
        print(''.join(tail[-12:]))
    return iters, peak_mem, oom, crashed

def summarize(iters, warmup):
    steady = iters[warmup:] if len(iters) > warmup else []
    if not steady:
        return None
    return dict(
        median=statistics.median(steady),
        mean=statistics.mean(steady),
        e2e_total=sum(iters),          # full end-to-end (all iters incl warmup)
        n=len(steady),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(CONFIGS))
    ap.add_argument('--gbs', type=int, default=8)
    ap.add_argument('--mbs', type=int, default=1)
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--pp', type=int, default=4)
    ap.add_argument('--gpus', type=int, default=4)
    ap.add_argument('--vp', type=int, default=2,
                    help='number of virtual pipeline stages (model chunks) for VP policies')
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--measure', type=int, default=100)
    ap.add_argument('--script', default='pretrain_gpt.py')
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--vocab-file', required=True)
    ap.add_argument('--merge-file', required=True)
    ap.add_argument('--policies', nargs='+', default=list(POLICIES))
    a = ap.parse_args()

    rows = []
    for pol in a.policies:
        print(f"\n{'='*60}\n[RUN] {pol}  ({a.config}, GBS={a.gbs}, MBS={a.mbs}, seq={a.seq})\n{'='*60}", flush=True)
        iters, mem, oom, crashed = run_one(pol, a)
        s = summarize(iters, a.warmup)
        rows.append((pol, s, mem, oom, crashed))

    print(f"\n\n{'='*78}")
    print(f"RESULTS  config={a.config} GBS={a.gbs} MBS={a.mbs} seq={a.seq} PP={a.pp}")
    print('='*78)
    print(f"{'policy':<18}{'median_ms':>12}{'mean_ms':>12}{'e2e_s':>10}{'peak_MB':>10}{'status':>10}")
    print('-'*78)
    for pol, s, mem, oom, crashed in rows:
        if oom:
            print(f"{pol:<18}{'-':>12}{'-':>12}{'-':>10}{'-':>10}{'OOM':>10}")
        elif crashed or s is None:
            print(f"{pol:<18}{'-':>12}{'-':>12}{'-':>10}{'-':>10}{'ERROR':>10}")
        else:
            print(f"{pol:<18}{s['median']:>12.1f}{s['mean']:>12.1f}"
                  f"{s['e2e_total']/1000:>10.1f}{mem:>10.0f}{'OK':>10}")
    print('='*78)
    ok = [(p, s) for p, s, m, o, c in rows if s and not o and not c]
    if ok:
        best = min(ok, key=lambda x: x[1]['median'])
        print(f"optimal (min median latency, no OOM): {best[0]}  ({best[1]['median']:.1f} ms)")

if __name__ == '__main__':
    main()
