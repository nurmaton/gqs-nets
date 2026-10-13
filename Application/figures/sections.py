# Section figures of the three halls, seen along the rows of blocks, the three halls of each figure at one scale: the deployed positions (panels, the polylines
# of the vertices V_ij for each i, the midline with its ends on the foot lines, the knee walls, the widest floor strip with at
# least 2.0 m of clear height, and the dimension lines of the span, the rise, H_c and W_20) and the transport positions (in
# the well of the low-bed semi-trailer within the road limits for (i), within the minimum door opening of the 20-ft container
# for (ii) and (iii), with the width W and height H with the panels).
# Inputs: data/ (stored positions and panels, data/halls.json), the clear-height curve of clearheight.py, the node sizes of
# figures/label_sizes.json. Outputs (out/figures/): sections_deployed_base.pdf and sections_transport_base.pdf (geometry,
# ticks, axis labels; 300 dpi PNG previews) and sections_deployed.tex, sections_transport.tex (TikZ: labels and dimension
# lines over the base PDF, in metres of each panel; \input them with the base PDF on the graphics path, with the TikZ library
# arrows.meta), and out/figure_sections.json (every printed number and every end point of a dimension line). The frames are
# fitted to the labels of a layout pass (two-line values); the printed layer keeps their places and sets the values of H_c and
# W_20 in one line inside the green strip (see print_dep, print_tr).
# usage: python3 figures/sections.py            requires numpy, scipy, matplotlib (a few seconds)
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json, math, copy
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.ticker import MultipleLocator
from scipy.spatial import ConvexHull
import halls as HL
import clearheight as CH
import tikzlabels as TL

plt.rcParams.update(TL.CM_RC)
OUT = os.path.join(HL.OUT, 'figures')
os.makedirs(OUT, exist_ok=True)
NAMES = ('i', 'ii', 'iii')
FIG_W = TL.FIG_W
DOOR = '#b03a2e'
TITLE = {'i': '(i) low-bed trailer', 'ii': '(ii) container, loop', 'iii': '(iii) container, tube'}
OFF = 2.5                 # pt between an arrow and the node border of its label
TITLE_GAP = 3.0           # pt between the top of the frame and the panel title
FEET_GAP = 2.5            # pt between the span node and the 'feet at' node
WELL_GAP = 2.5            # pt between the top edge of the well and its node
MARGIN = 3.0              # pt between the outermost label node and the frame
HC_X = 3.5                # pt between the H_c arrow and its node
HC_GAP = 3.0              # pt above and below the H_c node when it is centred between the floor and the W_20 node
HC_BLOCK_Y = 0.65         # m: centre height of the H_c node when the W_20 node lies outside the hall
EDGE = 1.5                # pt between the outermost text or frame and the border of the picture
INSEP = 1.0 * TL.BP       # inner sep of the nodes (1 pt TeX) in bp
LW = [(1.2, 1.5), (1.1, 1.5)]                 # midline, ground line and door outline: printed 1.5 bp
STY = {'dep': r'lab/.style={%s},greenlab/.style={lab,text=appgreen},greylab/.style={lab,text=appgrey},%%' % TL.LAB_STYLE + '\n  ' +
              TL.DIM_STYLE + ',' + TL.EXT_STYLE + ',' + TL.PIECE_STYLES,
       'tr': r'lab/.style={%s},redlab/.style={lab,text=appred},%%' % TL.LAB_STYLE + '\n  ' + TL.DIM_STYLE}
COL = {'dep': {'appgreen': (45, 106, 31), 'appgrey': (68, 68, 68), 'appzone': (0x4f, 0x9a, 0x3c)}, 'tr': {'appred': (176, 58, 46)}}
ROAD_LW = 1.5
W_LEVEL = -0.28           # m: the W arrow of the transport row (all three in the printed layer)
# drawing colours
C_COL, C_MID, C_SLAB, C_SLABE = '#8aa0b8', '#1f3a63', '#dde4ec', '#b4c1cf'
C_ZONE, C_ZONEE, C_WALL, C_GROUND = '#cfe9c4', '#4f9a3c', '#c9c9c9', '#6b6b6b'
SP = HL.SPEC
TRAILER, ROAD = SP['trailer'], SP['road']
DOOR_WH = tuple(SP['container_20ft']['door_min_W_H_m'])


def f2(v):
    return '%.2f' % (v + 1e-9)


# ------------------------------------------------------------------ data
def hull(pts):
    pts = np.asarray(pts, float)
    return pts[ConvexHull(pts).vertices]


MIRROR = {'iii': True}    # (iii) is seen from its other open end (x -> -x), as in the printed figure


def state(h, st):
    """one position seen along the rows: vertex polylines (one per i), midline, panel shadows; x = 0 midway between the feet"""
    V = HL.load_V(h, st); P = HL.load_panels(h, st)
    if MIRROR.get(h):
        V = V * [-1.0, 1.0, 1.0]; P = P * [-1.0, 1.0, 1.0]
    X = V[..., [0, 2]]
    mid = (X[1::2].mean(0) + X[0::2].mean(0)) / 2
    dx = -(mid[0, 0] + mid[-1, 0]) / 2
    return dict(cols=[X[i] + [dx, 0.0] for i in range(X.shape[0])], mid=mid + [dx, 0.0],
                slabs=[hull(np.unique(np.round(s[:, [0, 2]], 12), axis=0)) + [dx, 0.0] for s in P], t=HL.t_of(h, st))


def comfort(xs, h0, hw, half_span, wall=0.08):
    """H_c (largest clear height, at the point of the plateau of maxima nearest to x = 0) and the strip with at least 2.0 m
    of clear height around it, from the clear-height curve h0 above the foot line; floor hw below the foot line, bounded by
    the inner faces of the knee walls at |x| = half_span - wall when hw > 0"""
    lim = half_span - (wall if hw > 0 else 0.0)
    sel = (np.abs(xs) <= lim + 1e-9) & np.isfinite(h0)
    xs, hs = xs[sel], h0[sel] + hw
    k = int(np.nanargmax(hs)); Hc = float(hs[k])

    def run(thr):
        ok = hs >= thr
        if not ok[k]: return None
        a = k
        while a > 0 and ok[a - 1]: a -= 1
        b = k
        while b < len(ok) - 1 and ok[b + 1]: b += 1
        dx = np.median(np.diff(xs))
        return float(xs[a] - dx / 2), float(xs[b] + dx / 2)
    top = np.where(hs >= Hc - 1e-6)[0]
    xc = float(xs[top[np.argmin(np.abs(xs[top]))]])
    rr = CH.runs(hs >= 2.0)
    dx = np.median(np.diff(xs))
    runs20 = [(float(xs[a] - dx / 2), float(xs[b] + dx / 2)) for a, b in rr]
    return dict(Hc=Hc, xHc=xc, W20=run(2.0), xs=xs, hs=hs, runs20=runs20)


def prep(h):
    N = {'dep': state(h, 'deployed'), 'tr': state(h, 'transport')}
    mid = N['dep']['mid']
    span = float(np.linalg.norm(mid[-1] - mid[0])); rise = float(mid[:, 1].max() - mid[0, 1])
    hw = HL.HALLS[h]['feet_above_floor_m']
    Ps = HL.load_panels(h, 'deployed') * ([-1.0, 1.0, 1.0] if MIRROR.get(h) else 1.0)
    polys = CH.shadows(Ps); xg = CH.grid(polys); LO, HI = CH.vertical_extents(polys, xg)
    h0 = CH.clear_curve(LO, HI, xg, 0.0)
    cf = comfort(xg, h0, hw, span / 2)
    allp = np.vstack([N['tr']['cols'][0][:0]] + list(N['tr']['cols']) + list(N['tr']['slabs']))
    bx = (allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max())
    return dict(name=h, N=N, span=span, rise=rise, hw=hw, cf=cf, runs20=cf['runs20'], main=cf['W20'], bx=bx,
                door='truck' if h == 'i' else 'door', trW=bx[1] - bx[0], trH=bx[3] - bx[2])


# ------------------------------------------------------------------ drawing helpers
def section(ax, st, dx=0.0, dz=0.0, lw_mid=1.2, zorder=2):
    # round joins and caps: a mitre at a sharp corner would reach up to 2 bp past the drawn geometry
    for s in st['slabs']:
        ax.add_patch(Polygon(s + [dx, dz], closed=True, fc=C_SLAB, ec=C_SLABE, lw=0.25, joinstyle='round', zorder=zorder))
    for c in st['cols']:
        P = c + [dx, dz]
        ax.plot(P[:, 0], P[:, 1], color=C_COL, lw=0.4, solid_joinstyle='round', solid_capstyle='round', zorder=zorder + 1)
    M = st['mid'] + [dx, dz]
    ax.plot(M[:, 0], M[:, 1], color=C_MID, lw=lw_mid, solid_joinstyle='round', solid_capstyle='round', zorder=zorder + 2)


def ground(ax, x0, x1, y=0.0):
    ax.plot([x0, x1], [y, y], color=C_GROUND, lw=1.2, zorder=3)


def zone(ax, runs, top=2.0):
    for a, b in runs:
        ax.add_patch(Rectangle((a, 0.0), b - a, top, fc=C_ZONE, ec='none', alpha=0.9, zorder=1))
    if runs:
        a = min(r[0] for r in runs); b = max(r[1] for r in runs)
        ax.plot([a, b], [top, top], color=C_ZONEE, lw=0.7, ls='--', zorder=1.5)


def clip_rect(poly, x0, x1, y0, y1):
    """Sutherland-Hodgman clip of a polygon by an axis-aligned rectangle"""
    P = [tuple(p) for p in poly]
    for (axis, val, keep_ge) in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if not P: break
        Q = []
        for a, b in zip(P, P[1:] + P[:1]):
            ina = (a[axis] >= val) if keep_ge else (a[axis] <= val)
            inb = (b[axis] >= val) if keep_ge else (b[axis] <= val)
            if ina: Q.append(a)
            if ina != inb:
                s = (val - a[axis]) / (b[axis] - a[axis])
                Q.append((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))
        P = Q
    return np.array(P)


def place_box(st, x0=0.0, y0=0.0):
    """translation that puts the box of a position (panels included) on y0, centred at x0"""
    pts = np.concatenate([np.concatenate(st['cols'])] + [np.concatenate(st['slabs'])])
    b = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
    return x0 - (b[0] + b[1]) / 2, y0 - b[2], b


# ------------------------------------------------------------------ labels
def deployed_labels(nm, i, R):
    cf, hw = R['cf'], R['hw']
    a, b = R['main']
    L = [TL.Lab('dep%d_title' % i, i, TITLE[nm], 'lab', 'south west'),
         TL.Lab('dep%d_Hc' % i, i, r'$H_{\mathrm{c}}$\\%s~m' % ('%.2f' % cf['Hc']), 'lab', 'west', opts='align=left'),
         TL.Lab('dep%d_W20' % i, i, r'$W_{20}$\\%s~m' % ('%.2f' % (b - a)), 'greenlab', 'north', opts='align=center'),
         TL.Lab('dep%d_span' % i, i, r'span %s~m' % f2(R['span']), 'lab', 'north'),
         TL.Lab('dep%d_rise' % i, i, r'rise %s~m' % f2(R['rise']), 'lab', 'north', rot=90)]
    if hw > 0:
        L.append(TL.Lab('dep%d_feet' % i, i, r'feet at %s~m' % f2(hw), 'greylab', 'north'))
    return L


def transport_labels(nm, i, R):
    L = [TL.Lab('tr%d_title' % i, i, TITLE[nm], 'lab', 'south west'),
         TL.Lab('tr%d_W' % i, i, r'$W = %s$~m' % ('%.2f' % R['trW']), 'lab', 'north'),
         TL.Lab('tr%d_H' % i, i, r'$H = %s$~m' % ('%.2f' % R['trH']), 'lab', 'north', rot=90)]
    if R['door'] == 'truck':
        L.append(TL.Lab('tr%d_road' % i, i, r'road limits: width %.2f~m,\\height %.2f~m' % (ROAD['max_width_m'], ROAD['max_height_m']),
                        'redlab', 'south', opts='align=center'))
        L.append(TL.Lab('tr%d_well' % i, i, r'low-bed well,\\top at %s~m' % f2(TRAILER['well_top_above_road_m']), 'lab', 'south', opts='align=center'))
    else:
        L.append(TL.Lab('tr%d_door' % i, i, r'container door opening\\$%.3f \times %.3f$~m\\(ISO~668 minimum)' % DOOR_WH, 'redlab', 'south',
                        opts='align=center'))
    return L


# ------------------------------------------------------------------ deployed panels
def deployed(ax, R, xlim, ylim, ppm, panel):
    """base geometry of one deployed panel and its dimension lines (TikZ paths)"""
    N, hw = R['N'], R['hw']
    st = N['dep']; half = R['span'] / 2
    ground(ax, xlim[0], xlim[1])
    a, b = R['main']
    assert (a, b) in R['runs20'] and b - a == max(y_ - x_ for x_, y_ in R['runs20'])
    zone(ax, [(a, b)])                                    # only the widest strip, whose width W_20 is
    walls = []
    if hw > 0:
        allp = np.concatenate(st['slabs'])
        for sgn in (-1, 1):
            side = allp[sgn * allp[:, 0] > 0]; k = np.argmin(side[:, 1]); xl, zl = side[k]
            xa, xb = xl - 0.04, xl + 0.04
            ztop = min(clip_rect(sl, xa, xb, -5, 50)[:, 1].min() for sl in st['slabs'] if len(clip_rect(sl, xa, xb, -5, 50)) >= 3) + hw
            ax.add_patch(Rectangle((xa, 0.0), 0.08, ztop - 1e-4, fc=C_WALL, ec='k', lw=0.5, zorder=4))
            walls.append(dict(x=[float(xa), float(xb)], top=float(ztop - 1e-4)))
    section(ax, st, 0.0, hw, lw_mid=1.2)
    ax.plot([-half, half], [hw, hw], ls='none', marker='^', ms=3.5, color='k', zorder=9)
    cf = R['cf']
    ys = -0.26
    xr = max(np.concatenate(st['cols'])[:, 0].max(), max(s_[:, 0].max() for s_ in st['slabs'])) + 0.12
    top = hw + R['rise']
    # the crown of the midline: its highest vertex (the right one of a flat crown)
    kc = int(max(np.nonzero(st['mid'][:, 1] >= st['mid'][:, 1].max() - 1e-9)[0]))
    P = 'dep%d_' % panel
    dr = [TL.Draw(P + 'span_ext_left', panel, 'ext', (-half, ys), (-half, hw), what='span: extension line at the left foot'),
          TL.Draw(P + 'span_ext_right', panel, 'ext', (half, ys), (half, hw), what='span: extension line at the right foot'),
          TL.Draw(P + 'rise_ext_top', panel, 'ext', (st['mid'][kc, 0], top), (xr, top), what='rise: extension line from the crown of the midline')]
    if hw > 0:
        dr.append(TL.Draw(P + 'rise_ext_foot', panel, 'ext', (half, hw), (xr, hw), what='rise: extension line from the right foot'))
    else:
        dr.append(TL.Note(panel, 'rise: the foot line is the floor line here (feet on the floor); its extension line lies on the floor line: (%s,%s) -- (%s,%s)'
                          % (TL._f(half), TL._f(hw), TL._f(xr), TL._f(hw))))
    dr += [TL.Draw(P + 'Hc_arrow', panel, 'dim', (cf['xHc'], 0.0), (cf['xHc'], cf['Hc']), what='H_c: floor to the panels at the largest clear height'),
           TL.Draw(P + 'W20_arrow', panel, 'dim', (a, 2.0), (b, 2.0), color='appzone', what='W_20: ends of the widest strip at 2.0 m'),
           TL.Draw(P + 'span_arrow', panel, 'dim', (-half, ys), (half, ys), what='span: between the feet'),
           TL.Draw(P + 'rise_arrow', panel, 'dim', (xr, hw), (xr, top), what='rise: foot line to the crown of the midline')]
    return dict(xr=xr, ys=ys, hw=hw, top=top, a=a, b=b, xHc=cf['xHc'], Hc=cf['Hc'], xs=cf['xs'], hs=cf['hs'],
                slabs=[sl + [0.0, hw] for sl in st['slabs']], draws=dr, half=half, walls=walls)


def place_deployed(L, g, xlim, ylim, ppm):
    """positions of the labels of one deployed panel (distances in printed pt); returns where the W_20 node went"""
    lab = {l.key.split('_')[1]: l for l in L}
    lab['title'].place(xlim[0] - INSEP / ppm, ylim[1] + TITLE_GAP / ppm)
    # W_20: under its arrow, right of the H_c arrow, centred in the strip if it fits with 2.5 pt on each side; otherwise centred
    # between the H_c arrow and the inner surface of the panels, lowered in 0.5 pt steps until it clears both by 2.5 pt
    tw, th = lab['W20'], lab['Hc']
    tw.anchor = 'north'
    wm = tw.w / ppm; c = 2.5 / ppm; xs, hs, xHc, b = g['xs'], g['hs'], g['xHc'], g['b']

    def free_right(ytop):
        r = xs[(xs > xHc) & (hs < ytop + c)]
        return r.min() if len(r) else xs.max()
    d0 = OFF + 1.5
    if wm + 2 * c <= min(b, free_right(2.0 - d0 / ppm)) - xHc:
        tw.place(0.5 * (xHc + min(b, free_right(2.0 - d0 / ppm))), 2.0 - d0 / ppm)
    else:
        for d in np.arange(d0, 40.0, 0.5):
            xR = free_right(2.0 - d / ppm)
            if wm + 2 * c <= xR - xHc:
                break
        tw.place(0.5 * (xHc + xR), 2.0 - d / ppm)
    # H_c: HC_X pt right of its arrow, centred between the floor and the W_20 node, if it fits there with HC_GAP on both sides
    th.place(xHc + HC_X / ppm, 0.5 * (0.0 + tw.box(ppm)[1]))
    if tw.box(ppm)[1] - th.h / ppm >= 2 * HC_GAP / ppm:
        return 'inside'
    # otherwise: H_c at the centre height HC_BLOCK_Y, and W_20 outside the panels, above the left shoulder of the hall, as low as
    # the panels allow, MARGIN pt from the panels and from the frame
    th.place(xHc + HC_X / ppm, HC_BLOCK_Y)
    tw.anchor = 'south east'
    m = MARGIN / ppm
    for yb in np.arange(2.0, 4.0, 0.5 / ppm):
        parts = [clip_rect(sl, -50, 50, yb - m, yb + tw.h / ppm + m) for sl in g['slabs']]
        xo = min([p[:, 0].min() for p in parts if len(p) >= 3] + [np.inf])
        if xo - m - tw.w / ppm >= xlim[0] + m:
            break
    tw.place(xo - m, yb)
    return 'outside'


def place_rest(L, g, ppm):
    lab = {l.key.split('_')[1]: l for l in L}
    lab['span'].place(0.0, g['ys'] - OFF / ppm)
    lab['rise'].place(g['xr'] + OFF / ppm, (g['hw'] + g['top']) / 2)
    if 'feet' in lab:
        lab['feet'].place(0.0, lab['span'].box(ppm)[1] - FEET_GAP / ppm)


# ------------------------------------------------------------------ transport panels
def road_box(ax, x0, x1, y0, y1, ppm, color=DOOR, lw=ROAD_LW):
    """the road limits as a dashed box: four sides, each lw/2 past both corners, with a dash pattern scaled so that it starts and
    ends with a dash"""
    e = 0.5 * lw / ppm
    a0, g0 = 3.7 * lw, 1.6 * lw
    for P, Q in (((x0 - e, y0), (x1 + e, y0)), ((x1, y0 - e), (x1, y1 + e)), ((x1 + e, y1), (x0 - e, y1)), ((x0, y1 + e), (x0, y0 - e))):
        L = np.hypot(Q[0] - P[0], Q[1] - P[1]) * ppm
        n = int(round((L + g0) / (a0 + g0)))
        f = L / (n * a0 + (n - 1) * g0)
        ax.plot([P[0], Q[0]], [P[1], Q[1]], color=color, lw=lw, ls=(0, (a0 * f / lw, g0 * f / lw)), solid_capstyle='butt',
                dash_capstyle='butt', zorder=8)


def transport(ax, R, xlim, ylim, ppm, panel):
    st = R['N']['tr']
    ground(ax, xlim[0], xlim[1])
    g = {}
    if R['door'] == 'truck':
        deck = TRAILER['well_top_above_road_m']; wb = TRAILER['width_m']; wd = TRAILER['well_thickness_m']
        dx, dz, b = place_box(st, 0.0, deck)
        ax.add_patch(Rectangle((-wb / 2, deck - wd), wb, wd, fc='#9e9e9e', ec='k', lw=0.4, zorder=3))
        rw, rh = ROAD['max_width_m'], ROAD['max_height_m']
        road_box(ax, -rw / 2, rw / 2, 0.0, rh, ppm)
        g.update(rh=rh, deck=deck, road=(rw, rh))
        ylo = deck
    else:
        dx, dz, b = place_box(st, 0.0, 0.0)
        w, h = DOOR_WH
        ax.add_patch(Rectangle((-w / 2, 0.0), w, h, fill=False, ec=DOOR, lw=1.1, zorder=8))
        ax.add_patch(Rectangle((-w / 2, 0.0), w, h, fill=False, ec=DOOR, lw=0.6, ls=':', zorder=8))
        g.update(door_h=h)
        ylo = 0.0
    section(ax, st, dx, dz, lw_mid=1.2)
    fx = st['mid'][[0, -1], 0] + dx; fz = st['mid'][[0, -1], 1] + dz
    ax.plot(fx, fz, ls='none', marker='^', ms=3.5, color='k', zorder=9)
    W, H = b[1] - b[0], b[3] - b[2]
    yw = (ylo - 0.17) if R['door'] != 'truck' else W_LEVEL
    xh = max(W / 2, 1.17 if R['door'] != 'truck' else 1.275) + 0.10
    P = 'tr%d_' % panel
    g['draws'] = [TL.Draw(P + 'W_arrow', panel, 'dim', (-W / 2, yw), (W / 2, yw), what='W: width with the panels'),
                  TL.Draw(P + 'H_arrow', panel, 'dim', (xh, ylo), (xh, ylo + H), what='H: height with the panels')]
    g.update(yw=yw, xh=xh, ylo=ylo, H=H, W=W, box=(b[0] + dx, b[1] + dx, b[2] + dz, b[3] + dz))
    return g


def place_transport(L, g, xlim, ylim, ppm):
    lab = {l.key.split('_')[1]: l for l in L}
    lab['title'].place(xlim[0] - INSEP / ppm, ylim[1] + TITLE_GAP / ppm)
    lab['W'].place(0.0, g['yw'] - OFF / ppm)
    lab['H'].place(g['xh'] + OFF / ppm, g['ylo'] + g['H'] / 2)
    if 'road' in lab:
        lab['road'].place(0.0, g['rh'] + (OFF + 0.5) / ppm)
        lab['well'].place(0.0, g['deck'] + WELL_GAP / ppm)
    if 'door' in lab:
        lab['door'].place(0.0, g['door_h'] + (OFF + 1.0) / ppm)


# ------------------------------------------------------------------ printed label layer
# The frames are fitted (fit) to the label boxes of the layout pass, the labels of deployed_labels and transport_labels with the
# sizes of label_sizes.json. The printed layer keeps the places of that pass and its dimension lines, and changes: the extension
# lines are dashed with a pattern fitted to each line and drawn from the hall to the dimension line; H_c and W_20 are set in one
# line at 7 pt, centred in the green strip (W_20 under its arrow, H_c midway between the floor line and the W_20 node), and the H_c
# arrow is interrupted behind both; the W_20 arrow is 1 bp wide (it lies on the 1 bp dashed top edge of the strip); the W arrows of
# the transport row are at one level; the road limits, the door and the well get short labels. Lengths are in TeX pt, from the
# rounded maps written to the .tex file.
PT_BP = 72.27 / 72.0                     # TeX pt per bp
VAL_FONT = r'font=\fontsize{7}{8.5}\selectfont'
GAP = 0.5                                # pt between a node border and the interrupted H_c arrow


def print_dep(labs, G, maps, fig_h_bp, SZ):
    """printed layer of the deployed row: labels and paths per panel"""
    hpt = fig_h_bp * PT_BP                                    # pt: printed height of the picture
    r4 = lambda v: round(float(v), TL.DIM_DEC) + 0.0          # the metre coordinates as written
    out, draws = [], {}
    for i in range(3):
        ox, oy, sx, sy = maps[i]
        ppm = sx * TL.LINEWIDTH_PT                            # pt per m (equal scales)
        D = {d.key.split('_', 1)[1]: d for d in G[i]['draws'] if isinstance(d, TL.Draw)}
        a, b = r4(D['W20_arrow'].p0[0]), r4(D['W20_arrow'].p1[0])
        xH, Hc = r4(D['Hc_arrow'].p0[0]), r4(D['Hc_arrow'].p1[1])
        xc = 0.5 * (a + b)
        hW = round(SZ['dep%d_W20_value' % i]['h'], 2) * PT_BP / ppm      # node heights, rounded to 0.01 bp as printed
        hH = round(SZ['dep%d_Hc_value' % i]['h'], 2) * PT_BP / ppm
        assert max(SZ['dep%d_W20_value' % i]['w'], SZ['dep%d_Hc_value' % i]['w']) * PT_BP / ppm <= b - a, 'value label wider than the strip'
        wtop = 2.0 - OFF / ppm; wbot = wtop - hW; g = GAP / ppm
        yc = 0.5 * (0.75 * PT_BP / ppm + wbot - g)            # between the top of the floor line (1.5 bp wide) and the W_20 node
        dr = []
        for d in G[i]['draws']:
            k = d.key.split('_', 1)[1] if isinstance(d, TL.Draw) else None
            if k is None:
                dr.append(d)
            elif d.style == 'ext':
                p0, p1 = tuple(map(r4, d.p0)), tuple(map(r4, d.p1))
                L = math.hypot((p1[0] - p0[0]) * sx * TL.LINEWIDTH_PT, (p1[1] - p0[1]) * sy * hpt)
                dd, n = TL.dash_fit(L)
                if k.startswith('span'):
                    p0, p1 = p1, p0                           # from the foot down to the dimension line
                dr.append(TL.Draw(d.key, i, 'ext', p0, p1, what=d.what, opts='dash pattern=on %.4fpt off %.4fpt' % (dd, dd / 2),
                                  note='(%.2f pt, %d dashes)' % (L, n)))
            elif k == 'Hc_arrow':
                segs = [('dimS', 0.0, yc - hH / 2 - g), ('dimN', yc + hH / 2 + g, wbot - g), ('dimE', wtop + g, Hc)]
                dr += [TL.Draw(d.key + '_%d' % (j + 1), i, st, (xH, y0), (xH, y1),
                               what='%s, piece %d of 3 (interrupted at the labels)' % (d.what, j + 1)) for j, (st, y0, y1) in enumerate(segs)]
            elif k == 'W20_arrow':
                dr.append(TL.Draw(d.key, i, 'dim', d.p0, d.p1, color='appzone', what=d.what, opts='line width=1bp'))
            else:
                dr.append(d)
        draws[i] = dr
        for l in [l for l in labs if l.panel == i]:
            k = l.key.split('_', 1)[1]
            if k == 'Hc':
                l = TL.Lab(l.key, i, r'$H_{\mathrm{c}} = %s$~m' % f2(Hc), 'lab', 'center', xc, yc, pre=VAL_FONT, dec=4)
            elif k == 'W20':
                l = TL.Lab(l.key, i, r'$W_{20} = %s$~m' % f2(b - a), 'greenlab', 'north', xc, wtop, pre=VAL_FONT, dec=4)
            out.append(l)
    return out, draws


def print_tr(labs, G, ppm):
    """printed layer of the transport row: the W arrows at W_LEVEL, short labels of the road limits, the door and the well"""
    out, draws = [], {}
    for i in range(3):
        dr = []
        for d in G[i]['draws']:
            if isinstance(d, TL.Draw) and d.key.endswith('W_arrow'):
                d = TL.Draw(d.key, i, d.style, (d.p0[0], W_LEVEL), (d.p1[0], W_LEVEL), what=d.what)
            dr.append(d)
        draws[i] = dr
        for l in [l for l in labs if l.panel == i]:
            k = l.key.split('_', 1)[1]; l = copy.copy(l)
            if k == 'W':
                l.place(0.0, W_LEVEL - OFF / ppm)
            elif k == 'road':
                l.text = r'$%.2f \times %.2f$~m\\(road limits)' % (ROAD['max_width_m'], ROAD['max_height_m'])
            elif k == 'door':
                l.text = r'$%.3f \times %.3f$~m\\(ISO~668 minimum)' % DOOR_WH
            elif k == 'well':
                l.text = r'well,\\top at %s~m' % f2(TRAILER['well_top_above_road_m'])
            out.append(l)
    return out, draws


# ------------------------------------------------------------------ figure layout
def style(ax, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
    ax.tick_params(length=2, pad=2.5)


def axes_row(xlims, ylim, mg, gap=0.07):
    """one row of axes at one scale; mg = (left, right, bottom, top) in in; returns fig, axes, k (in per m)"""
    left, right, bottom, top = mg
    xr = [b - a for a, b in xlims]
    k = (FIG_W - left - right - gap * (len(xr) - 1)) / sum(xr)
    h = k * (ylim[1] - ylim[0]); H = bottom + h + top
    fig = plt.figure(figsize=(FIG_W, H))
    axs, x = [], left
    for w in xr:
        ax = fig.add_axes([x / FIG_W, bottom / H, k * w / FIG_W, h / H], sharey=axs[0] if axs else None)
        axs.append(ax); x += k * w + gap
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)
    return fig, axs, k


def text_extents(fig):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); k = 72.0 / fig.dpi
    bbs = [ax.get_tightbbox(r) for ax in fig.axes]
    return (min(b.x0 for b in bbs) * k, min(b.y0 for b in bbs) * k, max(b.x1 for b in bbs) * k, max(b.y1 for b in bbs) * k)


def fit(build, place_all, state, label_bounds, n=20):
    """fixed point of the layout: margins and limits so that the labels keep their distances (corrections below 0.01 pt)"""
    for it in range(n):
        fig, axs, k, G = build(state)
        ppm = k * 72.0
        place_all(axs, G, state, ppm)
        x0, y0, x1, y1 = text_extents(fig)
        W, H = fig.get_figwidth() * 72, fig.get_figheight() * 72
        corr = label_bounds(state, ppm)
        tops = max(axs[l.panel].transData.transform([[0.0, l.box(ppm)[3]]])[0, 1] * 72.0 / fig.dpi
                   for l in state['labs'] if l.key.endswith('_title'))
        dmg = [(EDGE - x0) / 72, (x1 - (W - EDGE)) / 72, (EDGE - y0) / 72, (tops - (H - EDGE)) / 72]
        err = max([abs(v) * ppm for v in corr] + [abs(v) * 72 for v in dmg])
        if err < 0.01:
            return fig, axs, k, G, it + 1
        plt.close(fig)
        state['mg'] = [m + d for m, d in zip(state['mg'], dmg)]
        state['apply'](corr)
    raise RuntimeError('layout did not converge: %s' % err)


def finish(name, fig, axs, k, G, state, report):
    ppm = k * 72.0
    labs = state['labs']
    TL.set_lw(fig, LW)
    maps = [TL.panel_map(fig, ax) for ax in axs]
    fig.savefig(os.path.join(OUT, name + '_base.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(os.path.join(OUT, name + '_base_preview.png'), dpi=300)
    labs, draws = state['print'](labs, G, [TL.rounded_map(m) for m in maps], fig.get_figheight() * 72.0, ppm)
    panels = [('panel %s: metres -> picture' % TITLE[nm].split()[0], maps[i], [l for l in labs if l.panel == i], draws[i])
              for i, nm in enumerate(NAMES)]
    TL.write_tikz(os.path.join(OUT, name + '.tex'), name + '_base', state['sty'], state['col'], panels,
                  '%s: labels and dimension lines over %s_base.pdf; one scope per panel, coordinates in metres' % (name, name))
    report[name] = dict(fig_size_in=[fig.get_figwidth(), fig.get_figheight()], scale_pt_per_m=ppm,
                        labels=[dict(key=l.key, text=l.text, x_m=l.X, y_m=l.Y) for l in labs],
                        draws=[dict(key=d.key, p0_m=d.p0, p1_m=d.p1) for i in range(3) for d in draws[i] if isinstance(d, TL.Draw)])
    plt.close(fig)


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    HL.header('Section figures (deployed and transport positions)')
    RS = {nm: prep(nm) for nm in NAMES}
    report = {}
    SZ = TL.sizes('sections')
    labs = [l for i, nm in enumerate(NAMES) for l in deployed_labels(nm, i, RS[nm])]
    labs_tr = [l for i, nm in enumerate(NAMES) for l in transport_labels(nm, i, RS[nm])]
    for l in labs + labs_tr:
        l.size(SZ)
    xls = []
    for nm in NAMES:
        st = RS[nm]['N']['dep']
        allx = np.concatenate([np.concatenate(st['cols'])[:, 0], np.concatenate(st['slabs'])[:, 0]])
        xls.append([allx.min() - 0.08, allx.max() + 0.6])
    yd = [-0.9, max((np.concatenate(RS[nm]['N']['dep']['slabs'])[:, 1].max() + RS[nm]['hw']) for nm in RS) + 0.08]
    S = dict(xlims=xls, ylim=yd, mg=[0.33, 0.03, 0.32, 0.2], labs=labs, sty=STY['dep'], col=COL['dep'], ytop0=yd[1],
             print=lambda labs_, G_, maps_, h_, ppm_: print_dep(labs_, G_, maps_, h_, SZ))

    def build_dep(s):
        fig, axs, k = axes_row([tuple(x) for x in s['xlims']], tuple(s['ylim']), s['mg'])
        G = {}
        for i, (ax, nm, xl) in enumerate(zip(axs, NAMES, s['xlims'])):
            style(ax, xl, s['ylim']); ax.xaxis.set_major_locator(MultipleLocator(1)); ax.set_yticks([0, 1, 2, 3])
            G[i] = deployed(ax, RS[nm], xl, s['ylim'], k * 72.0, i)
        axs[0].set_ylabel('up (m)', labelpad=2)
        for ax in axs:
            ax.set_xlabel('across (m)', labelpad=2)
        return fig, axs, k, G

    def place_dep(axs, G, s, ppm):
        for i in range(3):
            Li = [l for l in s['labs'] if l.panel == i]
            place_deployed(Li, G[i], s['xlims'][i], s['ylim'], ppm)
            place_rest(Li, G[i], ppm)

    def bounds_dep(s, ppm):
        corr = []
        for i in range(3):
            rise = next(l for l in s['labs'] if l.key == 'dep%d_rise' % i)
            corr.append(s['xlims'][i][1] - (rise.box(ppm)[2] + MARGIN / ppm))
        low = min(l.box(ppm)[1] for l in s['labs'] if l.key.split('_')[1] in ('span', 'feet'))
        corr.append((low - MARGIN / ppm) - s['ylim'][0])
        hi = max([s['ytop0']] + [l.box(ppm)[3] + MARGIN / ppm for l in s['labs'] if l.key.endswith('_W20') and l.anchor == 'south east'])
        corr.append(s['ylim'][1] - hi)
        return corr

    def apply_dep(corr):
        for i in range(3):
            S['xlims'][i][1] -= corr[i]
        S['ylim'][0] += corr[3]; S['ylim'][1] -= corr[4]
    S['apply'] = apply_dep
    fig, axs, k, G, it = fit(build_dep, place_dep, S, bounds_dep)
    walls = {nm: G[i]['walls'] for i, nm in enumerate(NAMES)}
    finish('sections_deployed', fig, axs, k, G, S, report)
    # transport
    xl = [-ROAD['max_width_m'] / 2 - 0.12, ROAD['max_width_m'] / 2 + 0.4]
    yt = [-0.6, ROAD['max_height_m'] + 0.6]
    T = dict(xlims=[xl, xl, xl], ylim=yt, mg=[0.33, 0.03, 0.32, 0.2], labs=labs_tr, sty=STY['tr'], col=COL['tr'],
             print=lambda labs_, G_, maps_, h_, ppm_: print_tr(labs_, G_, ppm_))

    def build_tr(s):
        fig, axs, k = axes_row([tuple(x) for x in s['xlims']], tuple(s['ylim']), s['mg'])
        G = {}
        for i, (ax, nm) in enumerate(zip(axs, NAMES)):
            style(ax, s['xlims'][i], s['ylim']); ax.xaxis.set_major_locator(MultipleLocator(0.5)); ax.yaxis.set_major_locator(MultipleLocator(1))
            G[i] = transport(ax, RS[nm], s['xlims'][i], s['ylim'], k * 72.0, i)
        axs[0].set_ylabel('up (m)', labelpad=2)
        for ax in axs:
            ax.set_xlabel('across (m)', labelpad=2)
        return fig, axs, k, G

    def place_tr(axs, G, s, ppm):
        for i in range(3):
            place_transport([l for l in s['labs'] if l.panel == i], G[i], s['xlims'][i], s['ylim'], ppm)

    def bounds_tr(s, ppm):
        xr = max(l.box(ppm)[2] for l in s['labs'] if l.key.endswith('_H')) + MARGIN / ppm
        lo = min(l.box(ppm)[1] for l in s['labs'] if l.key.endswith('_W')) - MARGIN / ppm
        hi = max(l.box(ppm)[3] for l in s['labs'] if l.key.split('_')[1] in ('road', 'door')) + MARGIN / ppm
        return [s['xlims'][0][1] - xr, lo - s['ylim'][0], s['ylim'][1] - hi]

    def apply_tr(corr):
        T['xlims'][0][1] -= corr[0]
        T['ylim'][0] += corr[1]; T['ylim'][1] -= corr[2]
    T['apply'] = apply_tr
    fig, axs, k, G, it = fit(build_tr, place_tr, T, bounds_tr)
    finish('sections_transport', fig, axs, k, G, T, report)
    vals = {}
    for nm in NAMES:
        R = RS[nm]
        vals['%s.figure.Hc_m' % nm] = R['cf']['Hc']; vals['%s.figure.W20_m' % nm] = R['main'][1] - R['main'][0]
        vals['%s.figure.span_m' % nm] = R['span']; vals['%s.figure.rise_m' % nm] = R['rise']
        vals['%s.figure.W_transport_m' % nm] = R['trW']; vals['%s.figure.H_transport_m' % nm] = R['trH']
    report['knee_walls'] = walls
    HL.write_result('figure_sections', dict(values=vals, details=report))
    for nm in NAMES:
        print('(%s): span %.2f, rise %.2f, H_c %.2f (x = %.4f m), W_20 %.2f (x %.4f .. %.4f m); transport W %.2f, H %.2f'
              % (nm, RS[nm]['span'], RS[nm]['rise'], RS[nm]['cf']['Hc'], RS[nm]['cf']['xHc'], RS[nm]['main'][1] - RS[nm]['main'][0],
                 RS[nm]['main'][0], RS[nm]['main'][1], RS[nm]['trW'], RS[nm]['trH']))
    print('written out/figures/sections_deployed{_base.pdf,_base_preview.png,.tex}, sections_transport{...}, out/figure_sections.json')
