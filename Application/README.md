# Application: three transportable arch halls

This folder lets a reader rebuild the three arch halls (i), (ii) and (iii) and recheck every number and claim made about
them: the net data and both positions, the sizes, the clear height, the panels and their collision check along the motion,
the mobility counts, the period-two translation, the exchange of row types in (iii), the snow check with CalculiX, and the
figures and renderings. `run_quick.sh` runs all fast checks and prints PASS or FAIL for every quantity against
`expected_values.json`.

Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the
authors. Every script carries this note in its header.

## 1. Requirements

| what | version used | needed for |
|---|---|---|
| Python 3 with numpy, scipy, matplotlib | Python 3.11, numpy 2.3.4, scipy 1.16.3, matplotlib 3.10.7 | everything |
| `../gqs.py` (the tool of this supplement) | md5 `5b67bf1f19d85f93f47e72181c81955e` | building the nets (checks/) |
| CalculiX `ccx` 2.23 | 2.23 | FE runs (optional; the quick run uses one short run) |
| Blender 5.x | 5.2.1 | renderings (optional; the quick run builds and tests the scenes only) |
| pillow | any | the artefact scan of rendered images only |

The scripts load `../gqs.py` read-only and write no bytecode (`python3 -B`); set `GQS_PY` to use another path. They use
gqs.py functions that are not a documented interface (`check_parameters`, `assemble_net`, `vertex_face_angles`, `Block`,
`admissible_t_range`, `synchronize`, `build_net`, `self_intersections`, `vdata`, `BASE_EDGE`): a later version of gqs.py
may need adjustments; `checks/check_nets.py` prints the md5 of the gqs.py it loads. Nothing needs a TeX installation: the
label sizes of the figures are stored in `figures/label_sizes.json`.

## 2. How to run

```
sh run_quick.sh                       # all fast checks and the comparison, about 13 min
python3 checks/compare.py             # the comparison table again (out/compare.txt)
```

Environment: `PYTHON` (default `python3`), `GQS_PY` (default `../gqs.py`), `CCX` (default `ccx`), `BLENDER` (default
`blender`), `APP_OUT` (output folder, default `./out`), `SKIP_FE_SMOKE=1`, `SKIP_RENDER_TEST=1`. All outputs go to `out/`;
nothing in the folder is changed. Output: `out/run_quick.log` (all messages), `out/compare.txt` (one line per quantity:
PASS / FAIL / SKIP, key, hall, expected value and rule, measured value, quantity), `out/<check>.json` (the measured values).
The comparison rules: `equal`; `roundN` (the measured value rounded to N decimals equals the printed value); `tol`; `max`;
`range`. Items that need ccx or Blender are SKIPped when these are not found.

Measured on a laptop (Apple M-series, 4 threads for ccx); a run from a fresh copy gave 266 PASS, 0 FAIL, 0 SKIP in 13.9 min:

| step | script | time |
|---|---|---|
| net data | `checks/check_nets.py` | 30 s |
| sizes | `checks/check_dimensions.py` | 1 s |
| clear height | `checks/check_clear_height.py` | 1 s |
| mobility, period two | `checks/check_mobility.py` | 3 min |
| exchange of the types a and c | `checks/check_exchange.py` | 3 s |
| panels along the motion | `checks/check_panels.py i`, `ii`, `iii` | 3 min, 35 s, 3 min |
| FE: stored results | `fem/check_results.py` | 3 s |
| FE: all 61 input decks, md5 | `fem/make_decks.py --all` | 2 min |
| FE: smoke test, k = 4 | `fem/smoke_test.py` | 70 s, 3 GB of memory, 65 MB of ccx files in out/fe/work |
| figures | `figures/sections.py`, `figures/snow.py` | 6 s |
| renderings: scene build and test | `sh renderings/render.sh --test-only`, `renderings/summary.py` | 30 s |

## 3. The halls

`data/halls.json` defines each hall: the flat angles (alpha_1, beta_1, gamma_1, delta_1) of the base block, the signs
(e_1, e_2, e_3, e_4) = (1, -1, 1, -1) and e_0 = 1, the net size m x n, the base block (kappa_0, nu_0), the row types
rho_1, ..., rho_{n-2}, the faces F_ij of the hall, the exact t of both positions, one gqs.py unit in metres (the base edge),
the height of the feet above the floor, and the external constants with their sources (20-ft container of ISO 668, road
limits of Directive 96/53/EC, the low-bed semi-trailer).

| | (i) | (ii) | (iii) |
|---|---|---|---|
| angles (deg) | 36, 144, 72, 72 | 120, 60, 60, 60 | 60, 60, 120, 60 |
| m x n, (kappa_0, nu_0) | 20 x 20, (2, 1) | 13 x 19, (1, 1) | 20 x 27, (1, 1) |
| rho_1 .. rho_{n-2} | baaacabaaccabacaaa | bccccccabaccccccb | aaacabacacacccacacabacaaa |
| faces F_ij | 1 <= i <= 18, 2 <= j <= 18 | 1 <= i <= 11, 1 <= j <= 17 | 1 <= i <= 18, 1 <= j <= 25 |
| t deployed; transport (stored) | 0.08; 0.3199999976 | cot 54 deg; cot 35.5 deg | 0.15; 0.75 |

The transport position of (i) is stored at theta_1 = 144.510657 deg (t = 0.3199999976); t = 0.32 moves its vertices by
1.4e-8 m. The positions of (ii) are stored at theta_1 = 108 and 71 deg (t = 0.72654 and 1.40195); the rounded t of the
summary table (0.7265 and 1.4019) move them by up to 4.7e-4 m. `check_nets.py` reports both.

`halls.py` builds the hall faces of the m x n net with gqs.py at any t (`Net`), and defines the standing frame of a
position: y along the period-two translation V_{i+2,j} - V_ij (the mean direction of the rows of blocks), x along the foot
chord of the midline, z up; the midline is the mean of the polyline of the even-i vertices and that of the odd-i vertices,
seen along y. Deployed: x = 0 midway between the feet, z = 0 on the foot line; transport: the box of vertices and panels
centred, on z = 0.

## 4. What each check reproduces

### checks/check_nets.py: net data
Inputs: `../gqs.py`, `data/halls.json`, `data/hall_*_V.npy`. For each hall: the number of row types (n - 2); rho_nu =
rho_{n-1-nu} for (ii) and (iii); M = a_1 b_1 c_1 d_1 of the flat angles (gqs.py `vdata`): 5/4, 9/8, 9/8; the numbers of
panels and hinges (306, 577; 187, 346; 450, 857); that every face of each hall is a central face of its net, so that every
hinge is an edge of a central face; the typical panel edge (median edge: 0.61, 0.53, 0.33 m); t of both positions; both
positions rebuilt by gqs.py, scaled, against the stored vertices (rigid fit, residual about 1e-14 m) and the standing frame
of halls.py against the stored frame; the faces of the bare net checked for self-intersections (gqs.py
`self_intersections`) at 25 positions between the two values of t (none). The hinge lines against the panel corners in the
deployed position (the snow check reads the stresses outside a radius r of the corners): the hinge lengths (559 x 0.610 and
18 x 0.987 m; 331 x 0.532 and 15 x 1.065 m; 841 x 0.326 and 16 x 0.652 m), the largest distance from a point of a hinge line
to the nearest panel corner (0.493, 0.532, 0.326 m; exact: along a segment, the distance to the nearest corner is largest at
an end or where two corners are equally near), and the number of hinges with points farther than r from every corner: at
r = 0.2 m all hinges of (i) and (ii), and the 16 hinges of length 0.652 m of (iii) (V_{i,6}V_{i,7} and V_{i,20}V_{i,21}, odd
i = 3, ..., 17; its other 841 hinge lines lie within 0.2 m of the corners); at r = 0.15 m all hinges of the three halls.

### checks/check_dimensions.py: sizes
Inputs: `data/hall_*_V.npy`, `data/hall_*_panels.npy`, `data/halls.json`. The profile length l (length of the midline) in
both positions (7.81 / 8.06, 8.79 / 9.07, 7.48 / 7.70 m) and its change (+3.3, +3.1, +3.0 %); the mirror symmetry of the
profile (the two feet of (i) are mirror images); the deployed span, rise and length L along the hall (4.00 x 2.84 x 11.65,
4.50 x 2.54 x 5.19, 3.96 x 2.40 x 5.27 m); the transport width W, height H and length L with the panels (2.35 x 3.27 x
11.91, 2.15 x 1.76 x 5.72, 1.77 x 2.23 x 5.72 m); the panel thickness (80 mm); the outer top of the crown above the crown of
the midline (0.12, 0.11, 0.10 m); the lowest panel point below the foot line (0.05, 0.15, 0.05 m: there the halls rest on
80 mm plinths (i) or knee walls (ii), (iii)); the shape of the deployed profile: the root mean square of the distances of the midline
vertices from the best-fitting parabola with vertical axis and from the best-fitting circle (orthogonal distances, least
squares; 0.090 / 0.209, 0.202 / 0.219, 0.120 / 0.093 m), so that the profiles of (i) and (ii) are nearer a parabola and
that of (iii) nearer a circle (for (ii) by a small margin: its terraced crown fits neither closely); the gap between the feet
of the packed tube (iii) (35 mm, exact distance of convex panels);
(i) on the low-bed semi-trailer: 0.30 m well + H = 3.57 m above the road (limit 4.00 m), W within 2.55 m, the trailer
extended to 13.21 + 5.55 = 18.76 m, beyond 16.50 m and 12.00 m king pin to rear; (ii) and (iii): W x H within the door
2.286 x 2.261 m and L within the floor 5.867 m of the 20-ft container.

### checks/check_clear_height.py (and clearheight.py): clear height
Inputs: `data/hall_*_deployed_panels.npy`, `data/halls.json`. The panels seen along the rows give, at every x (0.5 mm
step), the lowest panel material above the floor; the feet stand 0.08, 0.30, 0.50 m above the floor; the plinths of (i) and
the knee walls of (ii), (iii) are 80 mm thick, centred under the lowest panel corners. H_c, the largest clear height (2.74,
2.71, 2.79 m); W_20, the width of the widest floor strip with at least 2.0 m of clear height (2.00, 2.29, 2.30 m; for (iii)
all strips together are 2.33 m); for (i) the lowest height of the feet that gives W_20 >= 2.0 m (0.0775 m, hence 0.08 m);
for (ii) the clear height under the flat part of the terraced crown (2.66 m), H_c being reached in the notches between the
terraces.

### checks/check_mobility.py: one degree of freedom, period two
Inputs: `../gqs.py`, `data/halls.json`, `data/hall_*_V.npy`. The compatibility matrix of the rigid panels (unknowns: angular
and linear velocity of every panel; per hinge, equal velocity at its midpoint and no relative rotation about the two axes
normal to it; a locked hinge also none about the hinge) at 9 equally spaced values of t from the deployed to the transport
position, both included: null space of dimension 7 (six rigid motions and one more), and 6 with one hinge locked (five
hinges, each at the deployed position, the middle sample and the transport position), in each hall; for (iii),
t = 1/sqrt(2) is left out of the samples and checked alone: null space 18, dihedral angle 0 at 204 of 857 hinges (the hinges
whose angle changes sign there); the smallest |dihedral angle| at the samples (not 0); the
period-two translation: shifting i by two moves every V_ij by one translation in both positions and at all samples (largest
deviation about 1e-14 m); the length of that translation grows during folding by 3, 14 and 12 % (1.1144 -> 1.1482,
0.8083 -> 0.9221, 0.5055 -> 0.5644 m).

### checks/check_exchange.py: rows of (iii) with the types a and c exchanged
Inputs: `../gqs.py`, `data/halls.json` (`rows_before_exchange`). The rows before the exchange differ from the final rows
only by exchanges of the types a and c (gqs.py writes the types as 'a'..'d') in rows nu = 4, 8, 10, 12, 14, 16, 18, 22; both nets have the same profile in both positions (even-i
and odd-i polylines and midline, to 3e-14 m); the skew in plan (spread of the mean position along the hall of the vertex
columns) falls from 2.73 to 0.63 m (deployed) and from 2.45 to 0.57 m (transport). The rows before the exchange are those
of the search that produced (iii); the search itself is not part of this folder.

### checks/check_panels.py: panels and collisions along the motion
Inputs: `../gqs.py`, `data/halls.json`, `data/motion_t.json` (the 365, 181 and 259 positions t of the motion),
`data/hall_*_panels.npy`. Each face carries a rigid 80 mm panel on its inner side; the hinges are the outer edges; at each
hinge both panels are bevelled by the bisector plane at the position of smallest inner angle. At every position all panel
pairs whose boxes come within 0.30 m are tested (separating axes; exact distance of the closest pairs). Panels of (i) with a
common vertex overlap: their corners are cut at 90 vertices (180 panels) by the plane through the vertex equidistant from
the two face planes, at the position of deepest overlap, and the check is repeated. Results: no overlapping pair along the
motion in any hall; the smallest separating-axis bound of the distance between panels without a common vertex (188 mm (i),
19.8 mm (ii), 35 mm (iii), all at the transport position); this bound is a lower bound of every distance, and the exact
distance of the pair with the smallest bound is 188, 22.8 and 35 mm, so the smallest distance of (ii) lies between 19.8 and
22.8 mm; the panels built here equal the stored panels at both ends (to 1.4e-8 m for (i), whose stored
transport position is at t = 0.3199999976, and exactly for (ii), (iii)).

### fem/: snow check with CalculiX
- `shellnet.py`: the shell model (S8R elements, 80 mm steel/polyurethane sandwich as an equivalent shell with core shear,
  hinges on the outer face sheet, supports, rails, spindle, point latch), the loads (self-weight; snow on a cylindrical roof:
  undrifted 0.8, drifted triangles 1.0 and 2.0, on the parts of the smoothed roof sloping by at most 60 deg), the limits
  (span/200, 280/1.1 MPa, wrinkling 0.5 (E_f E_c G_c)^(1/3) = 71.6 MPa, core shear 0.10 MPa), capacities, *BUCKLE, and the
  mechanism gate. `classify.py`: the buckling modes (rule A: local if the region with |u| >= 0.3 max is smaller than 2 % of
  the net or nothing farther than 0.5 m from the peak reaches half of it; rule B in fe.py: broad if at least 8 face centres
  move 30 % of the peak).
- `fe.py`: the three halls and two variants (V1: both feet anchored; V3: foot 2 on rails, one spindle, one point latch);
  `fe.py static HALL V K` runs the gate, the four load cases and all capacities, and gives for every limit the place where
  it is reached first (the stress point or node, its nearest net vertex and whether that is a support or a spindle end, the
  distance to the nearest hinge line and free edge, the position across the span from foot 1 (0) to foot 2 (1)); its output
  also keeps the load combinations and the capacities by radius in the form of the stored results. `fe.py buckle HALL V K
  CASE NMODES [--scale s]` runs *BUCKLE under ULS and classifies the modes. Inputs: `fem/input/hall_*_fe_V.npy` (the
  deployed outer skin in the frame of the runs; equal to `data/` up to a rigid motion, checked). Vertex indices (i, j) in
  fem/ are those of this grid: the hall vertex V_{i+1,j+2} for (i) and V_{i+1,j+1} for (ii), (iii).
- `compare_static.py RUN.json`: compares a static run of fe.py with the stored result of the same hall, variant and k
  (`results/hall_*/static_V*_k*.json`): snow totals, self-weight, first frequency, the largest displacements and stresses
  of every load combination, the capacities per limit, snow case and radius, and the values fe.py computes itself (113
  quantities); prints the largest relative difference.
- `make_decks.py`: regenerates the input decks of the reported runs (no ccx) and compares their md5 (without the title line)
  with `decks/MD5SUMS`: 61 decks (statics and buckling of (i) at k = 8 and 12, of (ii) at k = 12, buckling of (iii) at k = 8
  under 0.5 and 0.05 x ULS), all identical. The 16 decks that set a value of the summary table are shipped in `decks/` as
  run (their title line keeps the internal name of the run); the static decks of (iii) were not kept from its runs and are
  written by fe.py.
- `check_results.py`: reads the stored results of the full runs (`results/`) and gives every FE number: the capacities
  s_k,max with both feet anchored (0.60, 0.64, 0.68 kN/m2) and with spindle and lock (0.43, 0.04, 0.04) with the governing limit
  and snow case; where the governing limit is reached: for (i) with both feet anchored at a free edge of an open end (total
  displacement, undrifted snow), for (ii) next to a support (wrinkling 0.222 m from the support vertex (1, 17)), for (iii) on a
  hinge line (wrinkling at the middle of the 0.326 m hinge line (2, 9)-(2, 10), 0.163 m from its ends), for (i) with spindle and lock on the hinge line (5, 10)-(5, 11) on the side of the foot on the rails (0.65 of
  the way across from foot 1 to foot 2, 2.52 m above the feet; the rise is 2.84 m); lambda_1 (1.26, 3.12, 2.66; 1.51, 2.07,
  2.03); the local modes not counted (0.76 at a support of (ii), 0.57 at the end of the spindle of (i)) and that their
  factors fall with the mesh (the mode at the support vertex (11, 0) of (ii) under S2+: 2.57, 1.25, 0.81 at k = 4, 8, 12, the
  same mode by MAC > 0.97; the lowest at the end of the spindle of (i): 0.87 at k = 8, 0.57 at k = 12); the wrinkling
  capacity of (i) at r = 0.2 / 0.15 / 0.1 m (0.82 / 0.69 / 0.58); for (i) with spindle and lock, the capacity 0 when the 0.15 m
  around the latch is not left out (the factored self-weight alone exceeds the wrinkling limit 0.025 m from the latch); the
  change of the capacities from k = 8 to 12 (up to 10 %); the spread of the rail foot (19.6 and 17.7 mm under self-weight;
  85.8 and 75.5 mm under snow, the largest displacement under snow being at the foot on the rails); the vertical displacement
  of (iii) at k = 4 / 6 / 8 (12.66 / 13.03 / 13.26 mm) and its largest change from k = 6 to 8 over the four stored series
  (2.43 %); the overhang of the open ends of (i) (1.64 m); elements and element lengths (51, 44, 41 mm); the peak memory of
  ccx (the job ledgers give the maximum resident set size in GiB = 2^30 bytes in their column max_rss_GB: 13.4 GiB = 14.4 GB
  for (iii) at k = 8, 14.7 GiB = 15.7 GB for (i) at k = 12); h/b and mu_3; the section values. The places for (ii) and (iii)
  come from the reruns in `results/reruns`, the others from the stored results.
- `smoke_test.py`: one short ccx run, (ii) with both feet anchored at k = 4 (1-2 min): compare_static.py finds the stored
  k = 4 results (113 quantities) reproduced to the last digit.
- `results/`: per hall the module output and the post-processing of the static runs, the buckling summaries, every mode with
  both rules (`modes/`), the buckling factors as written by ccx (`dat/`), mesh comparisons, and the job ledgers (wall time,
  peak memory in GiB). The ccx result files (.frd, about 21 GB) are not included.
- `results/reruns/`: two static runs of fe.py made with this folder, both feet anchored, (ii) at k = 12 (7 min, peak memory
  7.7 GB) and (iii) at k = 8 (9 min, 14.0 GB; maximum resident set size by /usr/bin/time -l, 4 threads): the fe.py output,
  its console log and the comparison with the stored run by compare_static.py (113 quantities, identical).

Rerunning the full studies (one ccx job at a time): (i) k = 12, both variants with buckling, about 6.5 h and 15.7 GB
(14.7 GiB); (ii) k = 12 about 1.5 h and 9.2 GB (8.6 GiB); (iii) k = 8 about 1-1.5 h and 14.4 GB (13.4 GiB); for example
`python3 fem/fe.py static ii V1 12`, `python3 fem/fe.py buckle ii V1 12 S2- 90`, `python3 fem/fe.py buckle iii V1 8 S2+ 30
--scale 0.5`. A static run is compared with the stored one by `python3 fem/compare_static.py out/fe/<name>.json`; the
buckling runs write their factors and mode classes in another form than `results/` and are compared by hand (factors in
`results/hall_*/dat/`, classes in `results/hall_*/modes/`). ccx *BUCKLE returns the factors nearest to 1, not the lowest ones; the runs therefore ask for many modes (90 to
250 for (i), (ii)) or scale the load (0.5 for (iii)), and the lowest local factors are upper bounds.

### figures/
- `sections.py`: the section figures (deployed and transport positions at one scale, with span, rise, H_c, W_20, W and H)
  from `data/` and `clearheight.py`: a base PDF per figure and a TikZ file with the labels and dimension lines in metres.
  The frames are fitted to the label boxes of a layout pass with two-line values; the printed layer keeps their places and
  dimension lines and sets H_c and W_20 in one line at 7 pt inside the green strip, with the H_c arrow interrupted behind
  them, all lines 0.6 pt and the extension lines dashed with a pattern fitted to each line (a dash of about 4 pt, a gap of
  half of it, a full dash at both ends); the W_20 arrow is 1 bp wide, the width of the dashed top edge of the strip it lies on.
  Both TikZ files equal the printed ones apart from the file names in them, and both base PDFs are the printed files byte
  for byte (with the versions of section 1). Hall (iii) is drawn seen from its other open end, as printed.
- `snow.py`: the snow figure (profiles under self-weight and undrifted snow at mid-length and at the weaker open end,
  displacements x 20) from `figures/snow_data/` (FE vertex and face-grid displacements of the six runs); base PDF and TikZ
  labels, identical to the printed figure (the TikZ file apart from its file names; the base PDF pixel for pixel at 600 dpi).
- `tikzlabels.py`, `label_sizes.json`: the TikZ label layer and the node sizes as TeX sets them (\footnotesize; the one-line
  values at 7 pt).

### renderings/
- `make_geo.py`: the scene data from `data/` (panels as convex polytopes in OBJ files, the frame data, knee walls); the OBJ
  files equal those of the printed renderings.
- `scene.py` (Blender): the scene of one hall (deployed hall, transport position at the container or on the trailer, a
  1.80 m person); `test_scene.py`: the geometry test of the meshes Blender builds (panels against the stored positions,
  contacts, no penetration, props, t of the positions) and the comparison with the reference scene of the printed rendering
  (`reference/`); `render.sh`: scene, test, render (Cycles, 1024 samples, 3000 x 1200, seed 7), artefact scan, test.
- `cameras.json`: the camera and render settings of each image (az, el, lens 55 mm, sensor 36 mm, distance, target, location,
  rotation); `anchors/`: for each image, the end points of the dimension and extension lines of its printed label layer in
  the scene (m) and in the image (px), with the lengths and printed values of the dimension lines and their sources;
  `constants.json`: the sizes of the container and trailer with their sources.
- The quick run builds the three scenes and tests them (they equal the reference scenes exactly); a render takes tens of
  minutes and is not bit-reproducible (denoiser, Blender version).

## 5. Details behind the text

The text states the results in short form. The details behind them are listed here, each with its value and the script
that produces or checks it.

### The nets and the panels
- Searches: at one value of t the profile must follow a parabola or a circular arc of the given span, and at a second the
  net must fit the trailer or the container door. (i): a beam search that fixes rho_1, rho_2, ... one after another, then a
  local search; (ii) and (iii): enumeration of the palindromic sequences, rho_nu = rho_{n-1-nu}. The searches are not part
  of this folder (section 7); `checks/check_nets.py` confirms rho_nu = rho_{n-1-nu} for (ii) and (iii).
- Exchange in (iii): after its search, the types a and c were exchanged in the rows nu = 4, 8, 10, 12, 14, 16, 18, 22, which
  reduces the skew in plan from 2.73 to 0.63 m (deployed) and from 2.45 to 0.57 m (transport) and leaves the profile unchanged
  in both positions, to 3.1e-14 m (`checks/check_exchange.py`).
- Panels: the 80 mm panels lie on the inner side of the faces; at each hinge both panels are bevelled by the bisector plane at
  the position of smallest inner angle; the corners of the panels of (i) are cut at 90 vertices, 180 panels
  (`checks/check_panels.py`).

### Transport
- Trailer for (i): closed length 13.21 m plus extension 5.55 m, that is, 18.76 m; king pin to rear 17.55 m, with the king
  pin 1.21 m behind the front (assumed: the closed trailer then meets the 12.00 m limit exactly). Directive 96/53/EC limits the
  articulated vehicle to 16.50 m (Annex I, 1.1) and the distance from the king pin to the rear of the semi-trailer to 12.00 m
  (Annex I, 1.6) (`checks/check_dimensions.py`, `renderings/constants.json`).
- Beyond these limits, the vehicle may circulate on a special permit for indivisible loads (Article 4(3) of the Directive)
  or under a national derogation (Article 4(4)). Whether a folded hall is an indivisible load is not examined.
- The knee walls of (ii) and (iii) are plain walls and ship flat; the plinths of (i) are plain strips 0.03 m high.

### Clear height
- (ii): H_c = 2.71 m (2.7098 m) is reached in the notches between the terraces of its crown; under the flat parts of the
  crown, the clear height is 2.66 m (2.6590 m) (`checks/check_clear_height.py`).

### Mobility
- One degree of freedom: the compatibility matrix of the rigid panels has a null space of dimension 7 (the six rigid motions
  and one more) at 9 equally spaced values of t from the deployed to the transport position, both positions included, in
  each hall; with one hinge locked, its dimension is 6 (five hinges, each at the deployed position, the middle sample and the
  transport position, in each hall) (`checks/check_mobility.py`).
- Special position of (iii): t = 1/sqrt(2) = 0.7071, between its deployed (0.15) and transport (0.75) positions, so that its
  motion passes through it. There the dihedral angle is 0 at 204 of its 857 hinges (the hinges whose angle changes sign
  there; all others at least 70.53 deg), and the null space has dimension 18 (`checks/check_mobility.py`). Every hinge of a
  hall is an edge of a central face of its net (`checks/check_nets.py`), at which the flexion in closed form has nonzero
  dihedral angles; thus the closed form does not cover this position. Whether another motion branches off there, and whether
  the drive with one spindle passes through it, were not examined. None of the 9 samples above lies within 0.001 of this t.

### Drive
- The rows of blocks lengthen during folding by 3, 14 and 12 % ((i), (ii), (iii)): the period-two translation
  V_{i+2,j} - V_ij grows from 1.1144 to 1.1482 m, from 0.8083 to 0.9221 m, and from 0.5055 to 0.5644 m
  (`checks/check_mobility.py`).

### Snow check: model, loads and limits
- Panel: steel face sheets 0.6 mm (E_f = 210 GPa, Poisson's ratio 0.3, 7850 kg/m3, yield stress 280 MPa), polyurethane core
  78.8 mm (E_c = 4 MPa, G_c = 3.5 MPa, 40 kg/m3, shear strength 0.10 MPa); mass 12.57 kg/m2; line hinges on the outer face
  sheet, 40 mm from the mid-surface (`fem/shellnet.py`, `panel_section`; printed by `fem/check_results.py`).
- Snow: EN 1991-1-3:2003, Section 5.3.5 (cylindrical roofs) with its Figures 5.5 and 5.6; s_k = 1.0 kN/m2, exposure and
  thermal coefficients 1, the recommended values, no National Annex. The snow acts vertically per unit plan area (EN
  1991-1-3, Section 5) on the parts where the smoothed roof slopes by at most 60 deg. Shape coefficient: 0.8 undrifted (S1);
  drifted (S2+, S2-), two triangles with the peaks 0.5 mu_3 = 1.0 and mu_3 = 2.0, on either side in turn; mu_3 =
  min(0.2 + 10 h/b, 2.0) takes its recommended upper value 2.0 because h/b (0.688, 0.555, 0.591) exceeds 0.18
  (`fem/shellnet.py`; h/b and mu_3 printed by `fem/check_results.py`).
- Limits: under self-weight plus snow (SLS = G + S), the displacement, vertical or total, at most span/200. Under
  ULS = 1.35 G + 1.5 S: the von Mises stress in the face sheets at most 280/1.1 = 254.5 MPa, the face compression at most the
  wrinkling stress 0.5 (E_f E_c G_c)^(1/3) = 71.6 MPa, the core shear at most 0.10 MPa, and the buckling factor
  lambda_1 >= 1 (`fem/shellnet.py`, `fem/fe.py`).
- Mesh: each hall is modelled entirely, with quadratic shell elements (S8R), k = 12 per panel edge for (i) and (ii) and
  k = 8 for (iii): 44064, 26928 and 28800 elements, 51, 44 and 41 mm long on the typical panel edge. The peak memory of ccx is
  14.4 GB for (iii) at k = 8 and 15.7 GB for (i) at k = 12 (`fem/check_results.py`).
- Reading radius: the stresses are read outside a radius r of every net vertex (the panel corners; the supports and the
  spindle ends are net vertices), r = 0.2 m for (i) and (ii) and r = 0.15 m for (iii) (`fem/fe.py`). The point latch of
  the locked hinge sits at the middle of a hinge, not at a net vertex; for (i) with spindle and lock, the stresses within
  0.15 m of it are also left out (`fem/fe.py`, `latch_zone`). For (iii), at r = 0.2 m, 841 of its 857 hinge lines (those
  of length 0.326 m) lie entirely within 0.2 m of the panel corners; only the 16 hinges of length 0.652 m reach farther.
  At r = 0.15 m, every hinge line of the three halls has points farther than r from every corner
  (`checks/check_nets.py`).
- Effect of the radius: for (i) with both feet anchored, the wrinkling limit alone gives the capacities 0.82, 0.69 and
  0.58 kN/m2 at r = 0.2, 0.15 and 0.1 m (`fem/check_results.py`).

### Snow check: results
- Local modes not counted, for example 0.76 at the support vertex (11, 0) of (ii) under S2- (the mode at the same vertex
  under S2+ falls from 2.57 to 1.25 and 0.81 at k = 4, 8, 12) and 0.57 at the end of the spindle of (i) (0.87 at k = 8)
  (`fem/check_results.py`; vertex indices of fem/).
- Where the capacity is set, both feet anchored: (i) the total displacement under undrifted snow, on a free edge of an open
  end, which overhangs the last supported vertex by up to 1.64 m; (ii) the wrinkling under drifted snow next to a support,
  0.222 m from the support vertex (1, 17); (iii) the wrinkling under drifted snow on a hinge line, at the middle of the
  0.326 m hinge line (2, 9)-(2, 10), 0.163 m from its ends. With spindle and lock, (i): the wrinkling under drifted snow on
  the hinge line (5, 10)-(5, 11), on the side of the foot on the rails (0.65 of the way across from foot 1 to foot 2,
  2.52 m above the feet; the rise is 2.84 m) (`fem/check_results.py`; vertex indices of fem/).

### Section figures and renderings
- Deployed section figure: the widest floor strip with at least 2.0 m of clear height is drawn in green up to 2.0 m; the
  plinths of (i), 0.03 m high, lie within the floor line (`figures/sections.py`).
- Renderings: Blender 5.2.1, Cycles (`renderings/render.sh`); the panel edges are drawn as lines, none behind the container
  walls (`renderings/scene.py`).
- The rise is measured to the crown of the midline; the outer top of the panels lies 0.12, 0.11 and 0.10 m above it
  (0.1240, 0.1109, 0.1030 m). The deployed length is measured over the stepped open ends: 11.65, 5.19, 5.27 m
  (`checks/check_dimensions.py`).
- Trailer (Kaessbohrer K.SLL 2/2H-18/20, extended, parked on landing legs, without tractor): width 2.54 m, well top 0.30 m
  and well length 6.81 + 5.55 m from the maker's product page; well thickness 0.20 m and tyre size 245/70 R17.5 from the
  maker's test article; deck height over the axles 0.95 m from its German version; tyre diameter 0.792 m from Michelin (XTA2
  Energy); ground clearance 0.10 m = 0.30 - 0.20 m. The landing legs, the king pin, the gooseneck and the other parts are
  schematic (`renderings/constants.json`, `renderings/scene.py`).
- Container (20-ft, ISO 668 type 1CC): doorless and translucent, with the minimum internal size 5.867 x 2.33 x 2.35 m, the
  minimum door opening 2.286 x 2.261 m and the nominal external size 6.058 x 2.438 x 2.591 m; floor top 0.17 m, assumed
  for the picture; underframe schematic (`renderings/constants.json`, `renderings/scene.py`).
- Supports: hall (i) stands on plinths and (ii), (iii) on knee walls, 80 mm thick and centred under the lowest panel
  corners, whose tops lie 0.05, 0.15 and 0.05 m (0.0504, 0.1509, 0.0486 m) below the foot line, so that the plinths are
  0.03 m (0.0296 m) high; with triangular gaps (seats, not designed) under the feet (`checks/check_dimensions.py`, `checks/check_clear_height.py`,
  `renderings/scene.py`).
- The packed tube of (iii) leaves 35 mm (0.0353 m) between its feet, faces F(1, 1) and F(2, 25)
  (`checks/check_dimensions.py`).
- Sub-pixel offsets keep faces from coinciding: the ground and the floors lie 0.1 mm below what stands on them, parts of the
  container stand 0.5 to 2 mm apart or inside its nominal size, and touching parts of the trailer overlap by 0.4 to 1 mm, its
  rims 2 mm wider than the tyres on each side (`renderings/scene.py`).

## 6. Data and provenance

- `data/hall_*_{deployed,transport}_V.npy`: the vertices V_ij of the hall (i along the hall, j across), metres, standing
  frame. `data/hall_*_{deployed,transport}_panels.npy`: per face (i-major order of `halls.face_labels`) the points whose
  convex hull is its 80 mm panel (8 points, 9 for (i) with cut corners; repeated points pad). These are the positions and
  panels of the collision check; all scripts recompute them or check them against gqs.py.
- `data/motion_t.json`: the positions t of the motion used by the collision check.
- `fem/input/`: the deployed outer skin as the FE runs read it; (iii) is the 13-row sub-net of its runs extended by the exact
  two-row translation (equal to the gqs.py window to 1e-14 m).
- `fem/results/reruns/`: the two fe.py static reruns (see fem/).
- `expected_values.json`: every checked quantity with its printed value and rule (266 items).

## 7. What is not recomputed here

- The FE numbers come from the stored results of the full runs; the folder regenerates every input deck identically, the
  quick run reruns one short model (k = 4), and two full static runs ((ii) at k = 12, (iii) at k = 8, both feet anchored) were
  rerun with fe.py and equal the stored runs (`fem/results/reruns`); the other full studies, in particular the buckling runs,
  are not rerun (hours, 9-16 GB).
- The vertical displacements of (iii) at k = 6 are read from the stored mesh-series results of a separate run of the same
  model; no deck of that run is kept.
- The peak memory is read from the job ledgers of the runs (maximum resident set size in GiB, converted to GB).
- The searches that found the nets (beam search, local search, enumeration of palindromes) are not included; the nets are
  given and checked. Their targets (a parabola or a circular arc of the given span) are not rechecked; the words
  parabola-like and circle-like rest on the profile fits of check_dimensions.py.
- The renderings themselves are not recomputed by the quick run (see renderings/).

## 8. Contents

`MANIFEST.txt` lists every file with its size and md5. Top level: `run_quick.sh`, `expected_values.json`, `halls.py`,
`clearheight.py`, `README.md`; folders `data/`, `checks/`, `fem/`, `figures/`, `renderings/`.
