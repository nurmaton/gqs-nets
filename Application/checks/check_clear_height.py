# Clear height inside the deployed halls, from the stored 80 mm panels (data/), projected along the rows of blocks.
# h(x): the height above the floor of the lowest panel material above the floor at the position x across the hall (0 where
# panel material, a plinth or a knee wall reaches the floor); the feet stand at the height of data/halls.json above the floor,
# and the plinths of (i) and the knee walls of (ii) and (iii) are 80 mm thick, centred under the lowest panel corner of each
# side. Quantities: the largest clear height H_c; W_20, the width of the widest strip with h >= 2.0 m (and the total width of
# all such strips); for (i) the lowest height h_w of the feet with W_20 >= 2.0 m (hence 0.08 m in data/halls.json); for
# (ii) the clear height under the flat parts of the terraced crown (runs of constant h at least 0.10 m wide). x step 0.5 mm.
# usage: python3 checks/check_clear_height.py          output: out/check_clear_height.json
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import halls as HL
from clearheight import DX, shadows, grid, vertical_extents, knee_walls, clear_curve, strips

T0 = time.time()
LEVEL = HL.SPEC['clear_height_level_m']
WALL = HL.SPEC['knee_wall_thickness_m']
HL.header('Clear height of the deployed halls (stored panels)')
vals, det = {}, {}


for h in HL.HALL_KEYS:
    H = HL.HALLS[h]
    Vs = HL.load_V(h, 'deployed'); Ps = HL.load_panels(h, 'deployed')
    mid = HL.midline(Vs); xfoot = 0.5 * abs(mid[-1, 0] - mid[0, 0])
    polys = shadows(Ps)
    xg = grid(polys)
    LO, HI = vertical_extents(polys, xg)
    hw = H['feet_above_floor_m']
    walls = knee_walls(Ps, WALL)
    inner = np.abs(xg) < xfoot
    hx = clear_curve(LO, HI, xg, hw, walls)
    Hc = float(hx[inner & np.isfinite(hx)].max()); xHc = float(xg[inner][np.argmax(np.where(np.isfinite(hx[inner]), hx[inner], -1))])
    W20, W20tot, run20 = strips(hx, xg, LEVEL)
    vals['%s.Hc_m' % h] = Hc; vals['%s.W20_m' % h] = W20
    det['%s' % h] = dict(feet_above_floor_m=hw, walls_x=walls, Hc_x=xHc, W20_total_all_strips_m=W20tot, W20_strip_x=run20)
    print('\nhall (%s): feet %.2f m above the floor%s: H_c = %.4f m (at x = %.3f m); W_20 = %.4f m (widest strip, x %.4f .. %.4f m); '
          'all strips together %.4f m' % (h, hw, (', %s x %.3f..%.3f and %.3f..%.3f m' % (('plinths' if h == 'i' else 'knee walls',) + walls[0] + walls[1])) if hw > 0 else '',
                                          Hc, xHc, W20, run20[0], run20[1], W20tot))
    if h == 'i':
        def w20(hw_):
            return strips(clear_curve(LO, HI, xg, hw_, walls), xg, LEVEL)[0]
        a, b = 0.0, 0.5
        for _ in range(40):
            c = 0.5 * (a + b)
            if w20(c) >= 2.0: b = c
            else: a = c
        vals['i.lowest_feet_height_for_W20_2m_m'] = b
        print('  W_20 >= 2.0 m from feet h_w = %.4f m above the floor on (feet at %.2f m: W_20 = %.4f m)' % (b, hw, W20))
    if h == 'ii':
        # flat parts of the terraced crown: runs of h within 1 mm of constant, at least 0.10 m wide, above 2.5 m
        cen = inner & np.isfinite(hx) & (hx > 2.5)
        levels = []
        k = 0; idx = np.where(cen)[0]
        a = 0
        while a < len(xg):
            if not cen[a]: a += 1; continue
            b = a
            while b + 1 < len(xg) and cen[b + 1] and abs(hx[b + 1] - hx[a]) < 1e-3: b += 1
            if (b - a + 1) * DX >= 0.10: levels.append((float(xg[a]), float(xg[b]), float(hx[a:b + 1].mean())))
            a = b + 1
        det['ii.crown_flat_parts'] = levels
        lv = sorted({round(l[2], 4) for l in levels})
        vals['ii.clear_height_flat_crown_m'] = float(min(l[2] for l in levels)) if levels else None
        print('  flat parts of the crown (x from, x to, h): %s' % ', '.join('%.3f..%.3f m: %.4f m' % l for l in levels))
        print('  clear height under the flat parts %.4f m; H_c %.4f m is reached in the notches between them' % (vals['ii.clear_height_flat_crown_m'], Hc))
path = HL.write_result('check_clear_height', dict(values=vals, details=det, seconds=time.time() - T0))
print('\nwritten %s (%.0f s)' % (os.path.relpath(path, HL.ROOT), time.time() - T0))
