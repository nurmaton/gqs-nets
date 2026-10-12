# Snow figure: profiles of the three halls under self-weight and undrifted snow (SLS = G + S1, s_k = 1 kN/m2), at mid-length
# and at the free edge of the weaker open end (the open end with the larger displacement with one spindle and one lock),
# undeformed and deformed with the displacements scaled by 20; top row both feet anchored, bottom row foot 2 on rails with one
# spindle and one locked hinge.
# Inputs: figures/snow_data/hall_*_V1|V3.npz (FE vertex positions Vpos and displacements Vu, and per face a 13 x 13 grid of
# points FX with displacements FU, at k = 12 for (i), (ii) and k = 8 for (iii)) with their .json (frame, feet), and the node
# sizes of figures/label_sizes.json. Mid-length profile: the undeformed surface cut by the plane y = c (c = middle of the hall
# length, to 0.01 m), each cut point moved by its interpolated displacement; open-end profile: the vertices of the free edge.
# Outputs (out/figures/): snow_profiles_base.pdf (+ 300 dpi PNG preview) and snow_profiles.tex (TikZ: panel titles and foot
# labels over the base PDF), snow_profiles.json (sections and extents).
# usage: python3 figures/snow.py              requires numpy, scipy, matplotlib (about 1 min)
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import halls as HL
import tikzlabels as TL

DATA = os.path.join(HERE, 'snow_data')
OUT = os.path.join(HL.OUT, 'figures')
os.makedirs(OUT, exist_ok=True)
SCALE = 20.0                      # displacement magnification
C_PIN, C_RAIL = '#2a78d6', '#e34948'          # foot 1 (anchored), foot 2
C_MID, C_END = '#4a3aa7', '#eb6834'           # deformed: mid-length, open end
C_UND = '#8f8e89'                             # undeformed
plt.rcParams.update(TL.CM_RC)
plt.rcParams.update({'axes.edgecolor': '#52514e'})
FIG_W = TL.FIG_W
HALLS = [('i', '(i)'), ('ii', '(ii)'), ('iii', '(iii)')]
ROWS = [('V1', 'both feet anchored'), ('V3', 'one spindle and one lock')]
MARGIN = 3.0              # pt between a foot node and the frame
FOOT_GAP = 3.5            # pt between the foot nodes and the lowest curve point above them
TITLE_GAP = 3.0           # pt between the frame and the title node
EDGE = 1.5                # pt between the outermost text or frame and the border of the picture
INSEP = 1.0 * TL.BP
LW = [(1.2, 1.5)]         # deformed profiles 1.2 -> 1.5; every width < 1 -> 1.0
STY = r'lab/.style={%s},footA/.style={lab,text=footblue},footB/.style={lab,text=footred}' % TL.LAB_STYLE
COL = {'footblue': tuple(int(C_PIN[i:i + 2], 16) for i in (1, 3, 5)), 'footred': tuple(int(C_RAIL[i:i + 2], 16) for i in (1, 3, 5))}


def load(h, var):
    z = np.load(os.path.join(DATA, 'hall_%s_%s.npz' % (h, var)))
    d = {k: z[k] for k in z.files}
    d['meta'] = json.load(open(os.path.join(DATA, 'hall_%s_%s.json' % (h, var))))
    return d


def end_columns(d):
    """vertex-column index of the low-y and of the high-y open end"""
    V = d['Vpos']
    return (0, V.shape[0] - 1) if V[0, :, 1].mean() < V[-1, :, 1].mean() else (V.shape[0] - 1, 0)


def plane_section(d, c):
    """the undeformed surface cut by the plane y = c: undeformed and deformed (x SCALE) (x, z) segments"""
    FX, FU = d['FX'], d['FU']
    nf, na, nb, _ = FX.shape
    tri = []
    for a in range(na - 1):
        for b in range(nb - 1):
            tri += [((a, b), (a + 1, b), (a + 1, b + 1)), ((a, b), (a + 1, b + 1), (a, b + 1))]
    TX = np.stack([np.stack([FX[:, p[0], p[1]] for p in t], 1) for t in tri], 1).reshape(-1, 3, 3)
    TU = np.stack([np.stack([FU[:, p[0], p[1]] for p in t], 1) for t in tri], 1).reshape(-1, 3, 3)
    s = TX[:, :, 1] - c
    s[np.abs(s) < 1e-12] = 1e-12
    px, pu, cnt = np.zeros((len(s), 2, 3)), np.zeros((len(s), 2, 3)), np.zeros(len(s), int)
    for e0, e1 in ((0, 1), (1, 2), (2, 0)):
        cr = s[:, e0] * s[:, e1] < 0
        t = np.where(cr, s[:, e0] / np.where(cr, s[:, e0] - s[:, e1], 1.0), 0.0)
        Px = TX[:, e0] + t[:, None] * (TX[:, e1] - TX[:, e0]); Pu = TU[:, e0] + t[:, None] * (TU[:, e1] - TU[:, e0])
        q = np.where(cr & (cnt < 2))[0]
        px[q, cnt[q]] = Px[q]; pu[q, cnt[q]] = Pu[q]; cnt[q] += 1
    ok = cnt == 2
    return px[ok][:, :, [0, 2]], (px[ok] + SCALE * pu[ok])[:, :, [0, 2]]


def edge_section(d, i):
    """the free edge of an open end (vertex column i) seen along the hall: undeformed and deformed (x SCALE) segments"""
    V, U = d['Vpos'][i], d['Vu'][i]
    P0 = V[:, [0, 2]]; P1 = (V + SCALE * U)[:, [0, 2]]
    return np.stack([P0[:-1], P0[1:]], 1), np.stack([P1[:-1], P1[1:]], 1), (V[:, 1].min(), V[:, 1].max())


def sections_of(d, spec):
    if spec[0] == 'plane': return plane_section(d, spec[1])
    s0, s1, _ = edge_section(d, spec[1]); return s0, s1


def chain(s0, s1, tol=1e-6):
    """join the cut segments into polylines (common end points within tol in the undeformed section)"""
    from scipy.spatial import cKDTree
    P0 = s0.reshape(-1, 2); P1 = s1.reshape(-1, 2)
    par = np.arange(len(P0))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in cKDTree(P0).query_pairs(tol):
        ra, rb = find(a), find(b)
        if ra != rb: par[max(ra, rb)] = min(ra, rb)
    key = np.array([find(a) for a in range(len(P0))])
    ids, inv = np.unique(key, return_inverse=True)
    X0 = np.zeros((len(ids), 2)); X1 = np.zeros((len(ids), 2)); cnt = np.zeros(len(ids))
    np.add.at(X0, inv, P0); np.add.at(X1, inv, P1); np.add.at(cnt, inv, 1)
    X0 /= cnt[:, None]; X1 /= cnt[:, None]
    E = inv.reshape(-1, 2)
    E = E[E[:, 0] != E[:, 1]]
    adj = {}
    for q, (a, b) in enumerate(E):
        adj.setdefault(a, []).append(q); adj.setdefault(b, []).append(q)
    used = np.zeros(len(E), bool)
    lines0, lines1 = [], []
    starts = [v for v, es in adj.items() if len(es) != 2] + list(adj.keys())
    for v0 in starts:
        for q0 in adj[v0]:
            if used[q0]: continue
            path, v, q = [v0], v0, q0
            while q is not None and not used[q]:
                used[q] = True
                a, b = E[q]; v = b if a == v else a; path.append(v)
                nxt = [e for e in adj[v] if not used[e]]
                q = nxt[0] if len(adj[v]) == 2 and nxt else None
            lines0.append(X0[path]); lines1.append(X1[path])
    return lines0, lines1


def common_sections(DD):
    """per hall: the plane at mid-length (to 0.01 m) and the free edge of the weaker open end (larger |u| in the V3 case)"""
    out, info = {}, {}
    for h in DD:
        d = DD[h]['V3']
        y = d['Vpos'][..., 1]
        c_mid = round(0.5 * (y.min() + y.max()), 2)
        i_lo, i_hi = end_columns(d)
        um = np.linalg.norm(d['Vu'], axis=-1)
        i_end = i_lo if um[i_lo].max() >= um[i_hi].max() else i_hi
        _, _, (ya, yb) = edge_section(d, i_end)
        out[h] = [('mid-length, y = %.1f m' % c_mid, ('plane', c_mid), C_MID, 'solid'),
                  ('open end (free edge), y = %.1f-%.1f m' % (ya, yb), ('edge', i_end), C_END, (0, (4, 2)))]
        info[h] = dict(mid_plane_y=c_mid, end_column=int(i_end), end_y=(float(ya), float(yb)),
                       umax_end_lo_mm=1e3 * float(um[i_lo].max()), umax_end_hi_mm=1e3 * float(um[i_hi].max()))
    return out, info


def panel(ax, d, secs, xlim, zlim):
    for (lab, spec, col, ls) in secs:
        s0, s1 = sections_of(d, spec)
        l0, l1 = chain(s0, s1)
        ax.add_collection(LineCollection(l0, colors=C_UND, linewidths=0.8, linestyles=ls, zorder=2, capstyle='round', joinstyle='round'))
        ax.add_collection(LineCollection(l1, colors=col, linewidths=1.2, linestyles='solid', zorder=3, capstyle='round', joinstyle='round'))
    ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect('equal', adjustable='box')
    ax.tick_params(length=2, pad=2.5)
    ax.xaxis.set_major_locator(MultipleLocator(2)); ax.yaxis.set_major_locator(MultipleLocator(1))


def lowest_under(ax, a, b):
    """lowest point of the drawn polylines of the axes over x in [a, b]"""
    lines = [np.asarray(p) for c in ax.collections for p in c.get_segments()]
    segs = np.concatenate([np.stack([p[:-1], p[1:]], 1) for p in lines if len(p) > 1])
    P = segs.reshape(-1, 2)
    zs = list(P[(P[:, 0] >= a) & (P[:, 0] <= b), 1])
    for s in segs:
        for xv in (a, b):
            if (s[0, 0] - xv) * (s[1, 0] - xv) < 0:
                zs.append(s[0, 1] + (xv - s[0, 0]) / (s[1, 0] - s[0, 0]) * (s[1, 1] - s[0, 1]))
    return min(zs)


def build(DD, secs, xlim, zlim, mg, labs, row_gap, leg_h, gap_x=0.07):
    left, right, bottom, top = mg
    k = (FIG_W - left - right - 2 * gap_x) / (3 * (xlim[1] - xlim[0]))
    h = k * (zlim[1] - zlim[0]); w = k * (xlim[1] - xlim[0])
    H = leg_h + bottom + 2 * h + row_gap + top
    fig = plt.figure(figsize=(FIG_W, H))
    axs = np.empty((2, 3), object)
    for r_ in range(2):
        for c_ in range(3):
            y0 = leg_h + bottom + (1 - r_) * (h + row_gap)
            axs[r_, c_] = fig.add_axes([(left + c_ * (w + gap_x)) / FIG_W, y0 / H, w / FIG_W, h / H],
                                       sharex=axs[0, 0] if (r_ or c_) else None, sharey=axs[0, 0] if (r_ or c_) else None)
    ppm = k * 72.0
    feet = {}
    for r_, (var, rname) in enumerate(ROWS):
        for c_, (hl, hname) in enumerate(HALLS):
            ax = axs[r_, c_]; p = 3 * r_ + c_
            panel(ax, DD[hl][var], secs[hl], xlim, zlim)
            ax.tick_params(labelbottom=(r_ == 1), labelleft=(c_ == 0))
            m = DD[hl][var]['meta']
            for lb in labs:
                if lb.panel != p: continue
                if lb.key.endswith('title'):
                    lb.place(xlim[0] - INSEP / ppm, zlim[1] + TITLE_GAP / ppm)
                    continue
                xf = m['x_foot1'] if lb.key.endswith('foot1') else m['x_foot2']
                wm = lb.w / ppm
                xc = min(max(xf, xlim[0] + wm / 2 + MARGIN / ppm), xlim[1] - wm / 2 - MARGIN / ppm)
                zlow = lowest_under(ax, xc - wm / 2 - 2 / ppm, xc + wm / 2 + 2 / ppm)
                feet.setdefault('%s_%s' % (hl, var), []).append(dict(text=lb.text, x_foot=float(xf), x_label=float(xc), lowest_curve_z_under_label=float(zlow)))
                lb.place(xc, zlow)
    zl = min(f['lowest_curve_z_under_label'] for v in feet.values() for f in v)
    for lb in labs:
        if not lb.key.endswith('title'):
            lb.place(lb.X, zl - FOOT_GAP / ppm)
    for ax in axs[1]:
        ax.set_xlabel('across (m)', labelpad=2)
    for ax in axs[:, 0]:
        ax.set_ylabel('up (m)', labelpad=2)
    hs = [Line2D([], [], color=C_UND, lw=0.8, label='undeformed, mid-length'),
          Line2D([], [], color=C_UND, lw=0.8, ls=(0, (4, 2)), label='undeformed, open end'),
          Line2D([], [], color=C_MID, lw=1.2, label=r'deformed $\times$20, mid-length'),
          Line2D([], [], color=C_END, lw=1.2, label=r'deformed $\times$20, open end')]
    leg = fig.legend(handles=hs, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.2,
                     columnspacing=2.0, borderaxespad=0.2)
    for l in leg.get_lines():
        l.set_linewidth(TL.LW_MIN if l.get_linewidth() < 1 else 1.5)
    return fig, axs, k, feet, leg


if __name__ == '__main__':
    HL.header('Snow figure: profiles under self-weight and undrifted snow, displacements x 20')
    DD = {h: {v: load(h, v) for v in ('V1', 'V3')} for h, _ in HALLS}
    secs, info = common_sections(DD)
    labs = []
    for r_, (var, rname) in enumerate(ROWS):
        for c_, (hl, hname) in enumerate(HALLS):
            p = 3 * r_ + c_
            labs.append(TL.Lab('sn%d_title' % p, p, '%s: %s' % (hname, rname), 'lab', 'south west'))
            labs.append(TL.Lab('sn%d_foot1' % p, p, 'foot 1', 'footA', 'north'))
            labs.append(TL.Lab('sn%d_foot2' % p, p, 'foot 2', 'footB', 'north'))
    SZ = TL.sizes('snow')
    for l in labs:
        l.size(SZ)
    xlim = (-2.5, 4.0)
    zlim = [-0.6, 3.1]
    mg = [0.33, 0.03, 0.30, 0.2]; row_gap, leg_h = 0.27, 0.36
    for it in range(30):
        fig, axs, k, feet, leg = build(DD, secs, xlim, tuple(zlim), mg, labs, row_gap, leg_h)
        ppm = k * 72.0
        fig.canvas.draw(); r = fig.canvas.get_renderer(); kk = 72.0 / fig.dpi
        W, H = fig.get_figwidth() * 72, fig.get_figheight() * 72
        low = min(l.box(ppm)[1] for l in labs if not l.key.endswith('title'))
        dz = (low - MARGIN / ppm) - zlim[0]
        tb = [ax.get_tightbbox(r) for ax in axs.flat]
        lb_ = leg.get_window_extent(r)
        x0 = min(min(b.x0 for b in tb), lb_.x0) * kk
        x1 = max(max(b.x1 for b in tb), lb_.x1) * kk
        top_title = max(axs[0, 0].transData.transform([[0, l.box(ppm)[3]]])[0, 1] for l in labs if l.panel < 3 and l.key.endswith('title')) * kk
        xl_bottom = min(ax.get_tightbbox(r).y0 for ax in axs[1]) * kk
        t2 = max(axs[1, 0].transData.transform([[0, l.box(ppm)[3]]])[0, 1] for l in labs if l.panel >= 3 and l.key.endswith('title')) * kk
        row1_bottom = axs[0, 0].get_window_extent(r).y0 * kk - 2.0 - 0.5
        dmg = [(EDGE - x0) / 72, (x1 - (W - EDGE)) / 72, 0.0, (top_title - (H - EDGE)) / 72]
        dgap = (lb_.y1 * kk + MARGIN - xl_bottom) / 72
        drow = (t2 + MARGIN - row1_bottom) / 72
        err = max(abs(dz) * ppm, *(abs(v) * 72 for v in dmg + [dgap, drow]))
        if err < 0.01:
            break
        plt.close(fig)
        zlim[0] += dz
        mg = [m + d for m, d in zip(mg, dmg)]
        leg_h += dgap
        row_gap += drow
    else:
        raise RuntimeError('layout did not converge: %s' % err)
    TL.set_lw(fig, LW)
    axmap = {3 * r_ + c_: axs[r_, c_] for r_ in range(2) for c_ in range(3)}
    maps = [TL.panel_map(fig, axmap[p]) for p in range(6)]
    fig.savefig(os.path.join(OUT, 'snow_profiles_base.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(os.path.join(OUT, 'snow_profiles_base_preview.png'), dpi=300)
    names = ['%s, %s' % (hn, rn) for rn in ('top row', 'bottom row') for _, hn in HALLS]
    panels = [('panel %s: metres -> picture' % names[p], maps[p], [l for l in labs if l.panel == p]) for p in range(6)]
    TL.write_tikz(os.path.join(OUT, 'snow_profiles.tex'), 'snow_profiles_base', STY, COL, panels,
                  'snow_profiles: labels over snow_profiles_base.pdf; one scope per panel, coordinates in metres')
    plt.close(fig)
    HL.write_result('figure_snow', dict(values={}, details=dict(sections=info, feet=feet, zlim=zlim, layout_iterations=it + 1)))
    for h, _ in HALLS:
        print('(%s): mid-length plane y = %.2f m; open end: vertex column %d (largest |u| with spindle and lock %.1f / %.1f mm at the two ends)'
              % (h, info[h]['mid_plane_y'], info[h]['end_column'], info[h]['umax_end_lo_mm'], info[h]['umax_end_hi_mm']))
    print('written out/figures/snow_profiles_base.pdf, snow_profiles_base_preview.png, snow_profiles.tex (%d layout iterations)' % (it + 1))
