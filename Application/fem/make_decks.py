# Regenerates the CalculiX input decks of the snow check (no ccx run) and compares them with the decks of the reported runs.
# fem/decks/MD5SUMS lists, for every deck of a reported case, the md5 of the deck without its title line (the line after
# *HEADING); the decks marked 'shipped' are in fem/decks as <name>.inp.gz. For each listed deck the model is rebuilt with
# fe.py (geometry fem/input, section, supports, spindle, latch, self-weight and snow loads) and the deck is written and
# compared. The shipped files are also checked against the list.
# usage: python3 fem/make_decks.py [--all | --shipped | NAME ...] [--keep DIR]
#        default --shipped (the 16 decks in fem/decks); --all: all 61 listed decks; --keep DIR keeps the written decks
# output: out/make_decks.json, one line per deck: same / DIFFERENT
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, re, gzip, json, time, hashlib, tempfile, shutil
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fe

T0 = time.time()


def body_md5(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as fh: L = fh.read().split('\n')
    assert L[0] == '*HEADING', path
    return hashlib.md5('\n'.join(L[:1] + L[2:]).encode()).hexdigest()


def parse(name):
    m = re.match(r'hall_(i|ii|iii)_(V1|V3)_k(\d+)_(static|buckle)(?:_c(\d))?(?:_(S1|S2p|S2m)_n(\d+))?(?:_s([\d.]+))?$', name)
    h, V, k, kind, c, case, n, s = m.groups()
    return dict(h=h, V=V, k=int(k), kind=kind, case={'S1': 'S1', 'S2p': 'S2+', 'S2m': 'S2-'}.get(case), nmodes=int(n) if n else None,
                scale=float(s) if s else 1.0)


lst = []
for line in open(os.path.join(HERE, 'decks', 'MD5SUMS')):
    if line.startswith('#') or not line.strip(): continue
    md, name, st = line.split()
    lst.append((name, md, st == 'shipped'))
args = [a for a in sys.argv[1:] if not a.startswith('--')]
keep = sys.argv[sys.argv.index('--keep') + 1] if '--keep' in sys.argv else None
if keep: args = [a for a in args if a != keep]
if args: todo = [x for x in lst if x[0] in args]
elif '--all' in sys.argv: todo = lst
else: todo = [x for x in lst if x[2]]
print('=' * 100); print('Input decks of the snow check: regeneration and md5 comparison (%d decks)' % len(todo)); print('=' * 100)
res = {}
# shipped files against the list
for name, md, ship in lst:
    if ship:
        p = os.path.join(HERE, 'decks', name + '.inp.gz')
        ok = os.path.exists(p) and body_md5(p) == md
        res['file:' + name] = ok
        if not ok: print('shipped file %s: DIFFERENT or missing' % name)
print('shipped deck files agree with MD5SUMS: %s' % all(v for k_, v in res.items() if k_.startswith('file:')))
groups = {}
for name, md, ship in todo:
    p = parse(name); groups.setdefault((p['h'], p['V'], p['k']), []).append((name, md, p))
work = keep or tempfile.mkdtemp(prefix='decks_')
os.makedirs(work, exist_ok=True)
for (h, V, k), items in groups.items():
    t1 = time.time()
    M = fe.make_model(h, V, k); G, snow, info = fe.loads(M, h)
    written = set()
    for name, md, p in items:
        if p['kind'] == 'static':
            if name not in written:
                written.update(fe.write_static_decks(M, G, snow, h, V, k, work))
        else:
            fe.write_buckle_deck(M, G, snow, h, V, k, p['case'], p['nmodes'], p['scale'], work)
        path = os.path.join(work, name + '.inp')
        got = body_md5(path)
        res[name] = got == md
        print('  %-40s %s  %s' % (name, got, 'same' if got == md else 'DIFFERENT (listed %s)' % md), flush=True)
        if not keep: os.remove(path)
    for nm in written:
        q = os.path.join(work, nm + '.inp')
        if not keep and os.path.exists(q): os.remove(q)
    print('  model hall (%s) %s k = %d: %d elements; snow totals S1 / S2+ / S2- = %.1f / %.1f / %.1f N, G = %.1f N (%.0f s)'
          % (h, V, k, len(M.elems), info['totals_N']['S1'], info['totals_N']['S2+'], info['totals_N']['S2-'], info['G_total_N'], time.time() - t1), flush=True)
if not keep: shutil.rmtree(work, ignore_errors=True)
n_same = sum(1 for k_, v in res.items() if not k_.startswith('file:') and v)
print('decks regenerated identically: %d of %d (%.0f s)' % (n_same, sum(1 for k_ in res if not k_.startswith('file:')), time.time() - T0))
out = os.path.join(os.environ.get('APP_OUT', os.path.join(os.path.dirname(HERE), 'out')), 'make_decks.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(dict(values={'fe.decks_regenerated_identically': bool(all(res.values()) and n_same > 0), 'fe.decks_checked': n_same},
               details=res, seconds=time.time() - T0), open(out, 'w'), indent=1)
