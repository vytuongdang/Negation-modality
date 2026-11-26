"""Re-derive V minus C0_box (violation among C0-mentioned), paired, pilot1000, 5 models.
Pure python. Image-level percentile bootstrap, 2000 resamples, seed 0.
Unit sets: 'row' = rows with asr_ok True (all 1000 pilot units);
'unit987' = additionally drop units with any asr_ok False row in scored_pilot1000_forms.jsonl (analyze.py-style unit asr_ok)."""
import json, random, sys, os, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
MODELS = ['qwen2.5-omni-7b', 'qwen2.5-omni-3b', 'minicpm-o-2.6', 'phi-4-mm', 'internomni']
T = lambda x: str(x) == 'True'

def load(p):
    return [json.loads(l) for l in open(p)]

def boot(items, B=2000, seed=0):
    # items: list of (image_id, diff)
    by = collections.defaultdict(list)
    for img, d in items: by[img].append(d)
    imgs = sorted(by); sums = [(sum(by[i]), len(by[i])) for i in imgs]
    rng = random.Random(seed); stats = []
    for _ in range(B):
        s = n = 0
        for _ in imgs:
            a, b = sums[rng.randrange(len(imgs))]; s += a; n += b
        stats.append(s / n)
    stats.sort()
    return stats[int(0.025 * B)], stats[int(0.975 * B) - 1]

for m in MODELS:
    a = load(f'{R}/{m}/scored_pilot1000.jsonl'); b = load(f'{R}/{m}/scored_pilot1000_C0_box-OCR_V.jsonl')
    bad = {r['unit_id'] for r in load(f'{R}/{m}/scored_pilot1000_forms.jsonl') if not T(r['asr_ok'])}
    c0 = {r['sample_id']: r for r in a if r['condition'] == 'C0'}
    v = {}
    for r in a:
        if r['condition'] == 'V': v.setdefault(r['sample_id'], []).append(r)
    box = {r['sample_id']: r for r in b if r['condition'] == 'C0_box'}
    assert all(len(x) == 1 for x in v.values())
    for uset in ('row', 'unit987'):
        for ref in ('C0', 'C0_box'):
            items = []
            for sid, rb in box.items():
                key = rb.get('pair_of') or sid
                rc, rv = c0[key], v[key][0]
                if not (T(rc['asr_ok']) and T(rv['asr_ok']) and T(rb['asr_ok'])): continue
                if uset == 'unit987' and rb['unit_id'] in bad: continue
                cond = rc if ref == 'C0' else rb
                if not T(cond['violation']): continue
                items.append((rb['image_id'], int(T(rv['violation'])) - int(T(rb['violation']))))
            n = len(items); d = 100 * sum(x for _, x in items) / n
            lo, hi = boot(items)
            print(f'{m}\t{uset}\tmentioned_by={ref}\tn={n}\tnimg={len({i for i,_ in items})}\tV-C0box={d:+.1f}\t[{100*lo:+.1f}, {100*hi:+.1f}]')
