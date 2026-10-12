# Summary of the scene tests of render.sh --test-only: for each hall, whether the geometry test passed and whether the scene
# built by Blender equals the reference scene of the printed rendering (every object and triangle within 1e-5 m).
# usage: python3 renderings/summary.py          output: out/render_test.json
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import halls as HL

C = os.path.join(HL.OUT, 'renderings', 'checks')
vals, det = {}, {}
for h in HL.HALL_KEYS:
    p = os.path.join(C, 'pre_hall_%s_two_test.json' % h)
    if not os.path.exists(p):
        vals['%s.render_scene_test_pass' % h] = False; continue
    d = json.load(open(p))
    ref = d.get('scene_reference') or {}
    vals['%s.render_scene_test_pass' % h] = bool(d.get('pass'))
    vals['%s.render_scene_equals_reference' % h] = bool(ref.get('file') and not ref.get('problems'))
    vals['%s.render_scene_worst_vertex_m' % h] = ref.get('worst_vertex_m')
    det[h] = dict(fails=d.get('fails'), states=d.get('states'), scene_reference=ref)
    print('hall (%s): geometry test %s; scene = reference scene of the printed rendering: %s (largest vertex distance %s m)'
          % (h, 'PASS' if d.get('pass') else 'FAIL', vals['%s.render_scene_equals_reference' % h], ref.get('worst_vertex_m')))
HL.write_result('render_test', dict(values=vals, details=det))
