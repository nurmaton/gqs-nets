# Compares the values measured by the checks (out/<check>.json) with expected_values.json and prints PASS / FAIL per quantity.
# Rules: equal (the same value), roundN (the measured value rounded to N decimals equals the expected one), tol (absolute
# tolerance: 1e-12 for M, 1e-6 m for the panel thickness), max (measured <= expected), range (within [lo, hi]). An item
# whose result file is missing FAILs, unless it is marked optional and its check was left out (then SKIP): run_quick.sh
# lists the checks it leaves out in out/skipped.txt; without that file every optional check counts as left out.
# usage: python3 checks/compare.py            exit code 1 if any item FAILs; summary in out/compare.json
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json
sys.dont_write_bytecode = True
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.environ.get('APP_OUT', os.path.join(ROOT, 'out'))
E = json.load(open(os.path.join(ROOT, 'expected_values.json')))['items']
cache = {}
SK = os.path.join(OUT, 'skipped.txt')
SKIPPED = set(open(SK).read().split()) if os.path.exists(SK) else None


def measured(check, key):
    if check not in cache:
        p = os.path.join(OUT, check + '.json')
        cache[check] = json.load(open(p))['values'] if os.path.exists(p) else None
    vals = cache[check]
    if vals is None: return 'missing file'
    return vals.get(key, 'missing value')


def judge(e, m):
    r, x = e['rule'], e['expected']
    if isinstance(m, str): return False
    if r == 'equal': return m == x
    if r.startswith('round'):
        n = int(r[5:]); return abs(round(float(m), n) - x) < 0.5 * 10 ** (-n - 3)
    if r == 'tol':
        tol = 1e-12 if '.M' in e['key'] else 1e-6
        return abs(float(m) - x) <= tol
    if r == 'max': return float(m) <= x
    if r == 'range': return x[0] <= float(m) <= x[1]
    raise ValueError(r)


def show(v):
    if isinstance(v, float):
        return '%.3e' % v if (v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5)) else '%.6g' % v
    return str(v)


n = dict(PASS=0, FAIL=0, SKIP=0)
rows = []
print('%-5s %-42s %-6s %-14s %-16s %s' % ('', 'key', 'hall', 'expected', 'measured', 'quantity'))
for e in E:
    m = measured(e['check'], e['key'])
    if m == 'missing file' and e.get('optional') and (SKIPPED is None or e['check'] in SKIPPED):
        st = 'SKIP'
    else:
        st = 'PASS' if judge(e, m) else 'FAIL'
    n[st] += 1
    exp = ('%s (%s)' % (e['expected'], e['rule'])) if e['rule'] not in ('equal',) else str(e['expected'])
    rows.append(dict(status=st, key=e['key'], expected=e['expected'], rule=e['rule'], measured=m, quantity=e['quantity']))
    print('%-5s %-42s %-6s %-14s %-16s %s%s' % (st, e['key'], e.get('hall') or '', exp[:14], show(m)[:16], e['quantity'],
                                               ('  [%s]' % e['optional']) if st == 'SKIP' else ''))
print('\n%d PASS, %d FAIL, %d SKIP of %d items' % (n['PASS'], n['FAIL'], n['SKIP'], len(E)))
json.dump(dict(counts=n, items=rows), open(os.path.join(OUT, 'compare.json'), 'w'), indent=1, default=str)
sys.exit(1 if n['FAIL'] else 0)
