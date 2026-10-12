# Hall (iii): the types a and c exchanged in some rows of blocks (gqs.py writes the types as 'a'..'d'). Both nets are built
# with ../gqs.py, from the types before the exchange (data/halls.json 'rows_before_exchange') and after it ('rows'), at the
# deployed and the transport t.
# Quantities: the rows in which the type differs; the profile of each net (the polylines of the vertices V_ij with even i and
# with odd i, and the midline, seen along the rows in the standing frame) and their largest difference between the two nets;
# the skew in plan (the spread of the mean position along the hall of the vertex columns j) of both nets.
# usage: python3 checks/check_exchange.py              output: out/check_exchange.json
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import halls as HL

T0 = time.time()
HL.header('Hall (iii): exchange of the types a and c in some rows of blocks')
h = 'iii'; H = HL.HALLS[h]; s = H['metres_per_unit']
before, after = H['rows_before_exchange'], H['rows']
diff = [k + 1 for k in range(len(after)) if before[k] != after[k]]
other = [k + 1 for k in range(len(after)) if before[k] == after[k] and before[k] not in 'ac']
print('rows before: %s\nrows after : %s\nchanged rows nu = %s (all %s); every change exchanges a and c: %s'
      % (before, after, diff, 'odd' if all(k % 2 for k in diff) else ('even' if not any(k % 2 for k in diff) else 'mixed'),
         all({before[k - 1], after[k - 1]} == {'a', 'c'} for k in diff)))
vals = {'iii.exchange_rows_changed': len(diff), 'iii.exchange_only_a_c': all({before[k - 1], after[k - 1]} == {'a', 'c'} for k in diff)}
det = {'changed_rows_nu': diff}
nets = {'before': HL.Net(h, rows=before), 'after': HL.Net(h, rows=after)}
worst = 0.0
for st in HL.STATES:
    t = H['t_%s' % st]
    prof, skew = {}, {}
    for key, net in nets.items():
        V, _ = net.hall_grid(t)
        Vs, _, _, _ = HL.to_standing(V * s, None, h, st)
        prof[key] = np.concatenate([Vs[0::2].mean(0)[:, [0, 2]], Vs[1::2].mean(0)[:, [0, 2]], HL.midline(Vs)])
        # spread of the polylines of each parity (all even-i polylines coincide when seen along the rows, so do the odd ones)
        spread = max(float(np.abs(Vs[p::2][:, :, [0, 2]] - Vs[p::2][:, :, [0, 2]].mean(0)).max()) for p in (0, 1))
        ymean = Vs[1:-1, :, 1].mean(0) if Vs.shape[0] % 2 == 1 else Vs[:, :, 1].mean(0)
        skew[key] = float(np.ptp(ymean))
        det['%s.%s' % (st, key)] = dict(parity_polyline_spread_m=spread, skew_in_plan_m=skew[key])
    d = float(np.abs(prof['before'] - prof['after']).max())
    worst = max(worst, d)
    vals['iii.exchange_profile_difference_%s_m' % st] = d
    vals['iii.skew_before_%s_m' % st] = skew['before']; vals['iii.skew_after_%s_m' % st] = skew['after']
    print('%-9s t = %.4f: profiles (even-i and odd-i polylines and the midline, standing frame) differ by at most %.1e m; '
          'skew in plan %.3f m before, %.3f m after' % (st, t, d, skew['before'], skew['after']))
vals['iii.skew_reduced'] = all(vals['iii.skew_after_%s_m' % st] < vals['iii.skew_before_%s_m' % st] for st in HL.STATES)
path = HL.write_result('check_exchange', dict(values=vals, details=det, seconds=time.time() - T0))
print('\nwritten %s (%.0f s)' % (os.path.relpath(path, HL.ROOT), time.time() - T0))
