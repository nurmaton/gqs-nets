# Short CalculiX run that shows the chain works: hall (ii), both feet anchored, k = 4 (2992 S8R elements, about 1-2 min
# and 3 GB of memory): mechanism gate, the four load cases, displacements, capacities and the places where the limits are
# reached (fe.py static), compared with the stored results of the same model (fem/results/hall_ii/static_V1_k4.json) by
# compare_static.py: snow totals, self-weight, first frequency, the largest displacements and stresses of every load
# combination, the capacities per limit, snow case and radius, and the displacements and capacities fe.py computes itself.
# usage: python3 fem/smoke_test.py [--threads N]      requires ccx 2.23 (CCX or 'ccx' on the PATH)
# output: out/fe_smoke.json (largest relative difference), work files in out/fe/work/
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fe
import compare_static as CS

T = int(sys.argv[sys.argv.index('--threads') + 1]) if '--threads' in sys.argv else 4
t0 = time.time()
print('=' * 100); print('CalculiX smoke test: hall (ii), both feet anchored, k = 4'); print('=' * 100)
a = fe.static('ii', 'V1', 4, T, None)
b = json.load(open(os.path.join(HERE, 'results', 'hall_ii', 'static_V1_k4.json')))
worst, pairs = CS.compare(a, b)
ok = bool(a['mechanism']['ok'])
print('mechanism gate passed: %s; largest relative difference to the stored k = 4 results: %.1e (%.0f s)' % (ok, worst, time.time() - t0))
out = os.path.join(fe.OUT, '..', 'fe_smoke.json')
json.dump(dict(values={'fe.smoke_gate_ok': ok, 'fe.smoke_max_rel_diff': worst}, details=[list(p) for p in pairs], seconds=time.time() - t0),
          open(out, 'w'), indent=1)
