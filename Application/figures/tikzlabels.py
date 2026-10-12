# TikZ label layer of the figures: the matplotlib scripts draw the base figure (geometry, ticks, axis labels, legend) at the
# printed width 6.151 in, so that it prints at scale 1; the texts and dimension lines become TikZ nodes and paths in a small
# .tex file, in the metre coordinates of each panel.
#   Lab         one label: text, style, anchor, metre position, rotation, and its node box in metres
#   Draw        one dimension arrow (style dim, or a piece of one: dimS, dimN, dimE) or extension line (style ext) in metres
#   panel_map   shift and scales of one panel (metres -> picture fraction) from the matplotlib transforms
#   write_tikz  the .tex file;  set_lw: every drawn line at least 1 bp
#   sizes       node sizes (bp) as TeX sets them, from label_sizes.json
# Units: matplotlib points are bp (1/72 in); TeX points are 1/72.27 in. Requires numpy and matplotlib.
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_W = 6.151                          # in: printed text width
BP = 72.0 / 72.27                      # bp per TeX pt
FS_TICK = 8.0 * BP                     # 8 pt (TeX) in bp: tick labels, axis labels, legend
LW_MIN = 1.0                           # bp: smallest line width
LAB_STYLE = r'font=\footnotesize,inner sep=1pt'
CM_RC = {'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
         'axes.unicode_minus': True, 'pdf.fonttype': 42, 'font.size': FS_TICK, 'xtick.labelsize': FS_TICK,
         'ytick.labelsize': FS_TICK, 'axes.labelsize': FS_TICK, 'legend.fontsize': FS_TICK, 'axes.linewidth': LW_MIN,
         'xtick.major.width': LW_MIN, 'ytick.major.width': LW_MIN, 'xtick.minor.width': LW_MIN, 'ytick.minor.width': LW_MIN,
         'axes.grid': False, 'lines.markeredgewidth': LW_MIN}


def sizes(figure):
    """node sizes {key: dict(w, h, base)} (bp) of the labels of one figure ('sections' or 'snow')"""
    with open(os.path.join(HERE, 'label_sizes.json')) as fh:
        return json.load(fh)[figure]


ANCH = {'south west': (0, 0), 'south': (.5, 0), 'south east': (1, 0), 'west': (0, .5), 'center': (.5, .5), 'east': (1, .5),
        'north west': (0, 1), 'north': (.5, 1), 'north east': (1, 1)}


class Lab:
    """one TikZ label: placed with its anchor at (X, Y) metres of its panel; rot 0 or 90 (counter-clockwise)"""
    def __init__(self, key, panel, text, style, anchor, X=0.0, Y=0.0, rot=0, opts='', pre='', dec=3):
        self.key, self.panel, self.text, self.style, self.anchor = key, panel, text, style, anchor
        self.X, self.Y, self.rot, self.opts = float(X), float(Y), rot, opts
        self.pre, self.dec = pre, dec       # options before the anchor (font); decimals of the position
        self.w = self.h = self.base = None

    def size(self, M):
        d = M[self.key]; self.w, self.h, self.base = d['w'], d['h'], d['base']
        return self

    def box(self, ppm):
        """node border box (x0, y0, x1, y1) in metres"""
        fx, fy = ANCH[self.anchor]
        w, h = self.w / ppm, self.h / ppm
        if self.rot == 0:
            x0, y0 = self.X - fx * w, self.Y - fy * h
            return (x0, y0, x0 + w, y0 + h)
        assert self.rot == 90
        sx, sy = self.X + fy * h, self.Y - fx * w
        return (sx - h, sy, sx, sy + w)

    def place(self, X, Y):
        self.X, self.Y = float(X), float(Y)
        return self

    def tikz(self):
        o = [self.style] + ([self.pre] if self.pre else []) + (['rotate=90'] if self.rot == 90 else []) + ['anchor=%s' % self.anchor] + \
            ([self.opts] if self.opts else [])
        return r'\node[%s] at (%.*f,%.*f) {%s};' % (','.join(o), self.dec, self.X + 0.0, self.dec, self.Y + 0.0, self.text)


def panel_map(fig, ax):
    """(ox, oy, sx, sy): picture fraction = (ox + sx X, oy + sy Y) for metres (X, Y) of the axes; checked to be affine"""
    T = lambda P: fig.transFigure.inverted().transform(ax.transData.transform(np.asarray(P, float)))
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    a, b = T([[x0, y0], [x1, y1]])
    sx, sy = (b[0] - a[0]) / (x1 - x0), (b[1] - a[1]) / (y1 - y0)
    ox, oy = a[0] - sx * x0, a[1] - sy * y0
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(x0 - 1, x1 + 1, 50), rng.uniform(y0 - 1, y1 + 1, 50)]
    err = np.abs(T(P) - np.c_[ox + sx * P[:, 0], oy + sy * P[:, 1]]).max()
    assert err < 1e-12, err
    return ox, oy, sx, sy


def rounded_map(m):
    return tuple(float('%.6f' % v) for v in m)


def write_tikz(path, base_name, styles, colors, panels, header):
    """panels: list of (comment, map, labels) or (comment, map, labels, draws); the paths of a panel are written first"""
    L = ['%% %s' % header]
    L += [r'\definecolor{%s}{RGB}{%s}%%' % (n, ','.join(str(v) for v in rgb)) for n, rgb in colors.items()]
    L.append(r'\begin{tikzpicture}[%s]' % styles)
    L.append(r'  \node[anchor=south west,inner sep=0,outer sep=0] (img) at (0,0) {\includegraphics[width=\linewidth]{%s}};' % base_name)
    L.append(r'  \begin{scope}[x={(img.south east)},y={(img.north west)}]   % picture coordinates 0..1')
    for P in panels:
        comment, m, labs = P[:3]
        draws = P[3] if len(P) > 3 else []
        ox, oy, sx, sy = rounded_map(m)
        L.append('    %% %s' % comment)
        L.append(r'    \begin{scope}[shift={(%.6f,%.6f)},xscale=%.6f,yscale=%.6f]' % (ox, oy, sx, sy))
        for d in draws:
            L.append('      ' + d.tikz())
        for lb in labs:
            L.append('      ' + lb.tikz())
        L.append(r'    \end{scope}')
    L.append(r'  \end{scope}')
    L.append(r'\end{tikzpicture}')
    open(path, 'w').write('\n'.join(L) + '\n')


# dimension arrows and extension lines: line width 0.6 pt; Latex tips 3.2 pt long and 2.8 pt wide; the pieces of an interrupted
# arrow carry the tip at their outer end only (dimS: at the start, dimE: at the end, dimN: none); extension lines dashed with a
# pattern fitted to each line (Draw.opts, see dash_fit)
DIM_TIP = r'Latex[length=3.2pt,width=2.8pt]'
DIM_STYLE = r'dim/.style={line width=0.6pt,{%s}-{%s}}' % (DIM_TIP, DIM_TIP)
EXT_STYLE = r'ext/.style={line width=0.6pt}'
PIECE_STYLES = r'dimS/.style={line width=0.6pt,{%s}-},dimE/.style={line width=0.6pt,-{%s}},dimN/.style={line width=0.6pt}' % (DIM_TIP, DIM_TIP)
DIM_DEC = 4                                 # decimals of the metre coordinates (0.1 mm)
LINEWIDTH_PT = 444.53284                    # pt: the printed width of the pictures (\linewidth, 6.151 in as TeX rounds it)


def dash_fit(L, d0=4.0):
    """dash d of a line L pt long: d close to d0, gap d/2, a full dash at both ends (L = n d + (n - 1) d/2, n >= 2 if L >= 4.5 pt)"""
    n = max(2 if L >= 4.5 else 1, round((L / d0 + 0.5) / 1.5))
    return L / (n + 0.5 * (n - 1)), n


def _f(v, n=DIM_DEC):
    return '%.*f' % (n, round(float(v), n) + 0.0)


class Draw:
    """one TikZ path from p0 to p1 (metres of its panel): style 'dim' (arrow) or 'ext' (extension line)"""
    def __init__(self, key, panel, style, p0, p1, color=None, what='', opts='', note=''):
        self.key, self.panel, self.style, self.color, self.what = key, panel, style, color, what
        self.opts, self.note = opts, note   # further options (dash pattern, line width); a note after the comment
        self.p0 = (float(p0[0]), float(p0[1])); self.p1 = (float(p1[0]), float(p1[1]))

    def tikz(self):
        o = self.style + (',' + self.color if self.color else '') + (',' + self.opts if self.opts else '')
        s = r'\draw[%s] (%s,%s) -- (%s,%s);' % (o, _f(self.p0[0]), _f(self.p0[1]), _f(self.p1[0]), _f(self.p1[1]))
        return s + ('   %% %s' % self.what if self.what else '') + ('   %s' % self.note if self.note else '')


class Note:
    """a comment line in a panel scope of the .tex file"""
    def __init__(self, panel, text):
        self.panel, self.text = panel, text

    def tikz(self):
        return '%% %s' % self.text


def set_lw(fig, table):
    """every drawn line at least LW_MIN: table maps a drawn width to its printed width (bp); other widths are raised to LW_MIN"""
    from matplotlib.collections import LineCollection
    from matplotlib.text import Annotation

    def new(w):
        for k, v in table:
            if abs(w - k) < 1e-9: return v
        return max(w, LW_MIN)
    for ax in fig.axes:
        for l in ax.lines:
            if l.get_linestyle() not in ('None', 'none', '', ' '): l.set_linewidth(new(l.get_linewidth()))
        for p in ax.patches:
            if p.get_edgecolor()[3] > 0 and p.get_linewidth() > 0: p.set_linewidth(new(p.get_linewidth()))
        for c in ax.collections:
            if isinstance(c, LineCollection): c.set_linewidths([new(v) for v in np.atleast_1d(c.get_linewidths())])
        for t in ax.texts:
            if isinstance(t, Annotation) and t.arrow_patch is not None: t.arrow_patch.set_linewidth(new(t.arrow_patch.get_linewidth()))
