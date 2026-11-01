# Application: three transportable arch halls

This folder holds the data of the arch halls (i), (ii), and (iii) and the scripts that rebuild them and check their
values in the paper. `MANIFEST.txt` lists every file with its size and md5.

## 1. Requirements

| Requirement | Version used | Needed for |
|---|---|---|
| Python 3 with numpy, scipy, matplotlib | Python 3.11, numpy 2.3.4, scipy 1.16.3, matplotlib 3.10.7 | Everything |
| `../gqs.py` | md5 `8f4661f1c6e4529e2c6cb6f6588e6b85` | Building the nets (`checks/`) |
| CalculiX `ccx` 2.23 | 2.23 | FE runs (optional) |
| Blender 5.x | 5.2.1 | Renderings (optional) |
| pillow | Any | The artefact scan of rendered images |

The scripts use internal functions of `../gqs.py`, so a later version may need adjustments. No TeX installation is
needed.

## 2. How to run

```
sh run_quick.sh                               # all fast checks and the comparison, about 13 min
python3 checks/compare.py > out/compare.txt   # the comparison table again
```

`run_quick.sh` compares the 272 quantities of `expected_values.json` with their measured values; a run from a fresh copy gave 272 PASS,
0 FAIL, 0 SKIP. All output goes to `out/`:

| File | Content |
|---|---|
| `out/compare.txt` | PASS, FAIL, or SKIP for each quantity |
| `out/run_quick.log` | All messages |
| `out/<check>.json` | The measured values |
| `out/skipped.txt` | The optional checks left out |

Environment variables and their defaults: `PYTHON=python3`, `GQS_PY=../gqs.py`, `CCX=ccx`, `BLENDER=blender`, and
`APP_OUT=./out`. `SKIP_FE_SMOKE=1` and `SKIP_RENDER_TEST=1` leave out the CalculiX and Blender steps, as does a missing `ccx` or `blender`; their quantities
are then marked SKIP.

## 3. The halls

| File | Content |
|---|---|
| `data/halls.json` | The three halls and the external constants, with their sources |
| `halls.py` | The hall faces at any $`t`$, and the standing frame: $`x`$ along the foot chord, $`y`$ along $`V_{i+2,j} - V_{ij}`$, $`z`$ up |
| `data/hall_*_{deployed,transport}_V.npy` | The vertices $`V_{ij}`$ ($`i`$ along the hall, $`j`$ across), in metres, in the standing frame |
| `data/hall_*_{deployed,transport}_panels.npy` | Per face (order of `halls.face_labels`), the points whose convex hull is its panel |
| `data/motion_t.json` | The values of $`t`$ at which the panels are checked |

For (i) in transport and for (ii), the stored $`t`$ differ slightly from those printed in the paper;
`checks/check_nets.py` reports the deviations.

## 4. The scripts

### checks/

| File | Content |
|---|---|
| `check_nets.py` | The nets rebuilt with `../gqs.py`: their data, self-intersections, and hinge lines |
| `check_dimensions.py` | The lengths, sizes, profile fits, and transport fits |
| `check_clear_height.py` | $`H_{\mathrm{c}}`$ and $`W_{20}`$, with `clearheight.py` |
| `check_mobility.py` | The mobility along the motion and the period-two translation |
| `check_exchange.py` | (iii) before and after the exchange of the types a and c |
| `check_panels.py` | The panels along the motion, per hall (`i`, `ii`, `iii`) |

### fem/

| File | Content |
|---|---|
| `shellnet.py` | The shell model, loads, limits, and capacities of the snow check |
| `classify.py` | The buckling modes classed as local or broad (global) |
| `fe.py` | Static and buckling runs of a hall and variant |
| `compare_static.py` | A static run of `fe.py` against the stored result |
| `make_decks.py` | The input decks of the reported runs (`--all`), written without ccx and compared with `decks/MD5SUMS` |
| `decks/` | 16 of these decks, as run |
| `check_results.py` | The FE values, from the stored results |
| `smoke_test.py` | One short ccx run, (ii) at $`k = 4`$, against the stored results |
| `results/` | The stored results, without the ccx .frd files |

V1: both feet anchored; V3: foot 2 on rails, one spindle, one point latch. S1: undrifted snow; S2+ and S2-: snow drifted
to either side. K: elements per panel edge ($`k`$). Vertex $`(i, j)`$ in `fem/` is the hall vertex $`V_{i+1,j+2}`$
for (i) and $`V_{i+1,j+1}`$ for (ii) and (iii).

A full study takes hours and 9 to 16 GB of memory, a single run minutes; for example:

```
python3 fem/fe.py static ii V1 12                     # fe.py static HALL V K
python3 fem/fe.py buckle ii V1 12 S2- 90              # fe.py buckle HALL V K CASE NMODES [--scale s]
python3 fem/fe.py buckle iii V1 8 S2+ 30 --scale 0.5
python3 fem/compare_static.py out/fe/hall_ii_V1_k12_static.json   # the first run against the stored one
```

Buckling runs are compared by hand with `fem/results/`. As ccx `*BUCKLE` returns the factors nearest to 1, not the
lowest ones, the runs ask for many modes or scale the load; the lowest local factors are upper bounds.

### figures/ and renderings/

| File | Content |
|---|---|
| `figures/sections.py` | The section figures |
| `figures/snow.py` | The snow figure, from `figures/snow_data/` |
| `figures/tikzlabels.py`, `figures/label_sizes.json` | The TikZ label layer and the label sizes |
| `renderings/make_geo.py` | The scene data from `data/` |
| `renderings/scene.py` | The Blender scene of one hall, deployed and in transport |
| `renderings/test_scene.py` | The geometry test of the Blender scenes and their comparison with `renderings/reference/` |
| `renderings/render.sh` | Scene, test, render, artefact scan, and test; `--test-only` stops before the render |
| `renderings/cameras.json`, `renderings/anchors/` | The camera and render settings and the dimension lines of each image |
| `renderings/constants.json` | The sizes of the container and the trailer, with their sources |

## 5. The values and their scripts

The tables give the script that produces or checks each value of Sections SM2.1 to SM2.4 and SM2.6 of the supplementary
materials.

### Nets and panels

| Value | Script |
|---|---|
| The data of the halls: $`M`$, panels, hinges, typical panel edge, $`t`$; $`\rho_\nu = \rho_{n-1-\nu}`$ for (ii) and (iii); no self-intersections of the bare nets | `data/halls.json`, `checks/check_nets.py` |
| The profile length $`\ell`$ and its change, span, rise, $`L`$, $`W`$, $`H`$; the parabola-like or circle-like profiles | `checks/check_dimensions.py` |
| The exchange of the types a and c in (iii), the unchanged profile, the skew in plan | `checks/check_exchange.py` |
| The panels: hinging, bevels, corner cuts of (i), no overlap along the motion, smallest distances (a lower bound and the exact distance of the pair with the smallest bound) | `checks/check_panels.py` |

### Transport and clear height

| Value | Script |
|---|---|
| The height of (i) above the road; the fits within the road limits, through the door, and on the container floor; the plinths | `checks/check_dimensions.py` |
| The trailer lengths and the length limits | `checks/check_dimensions.py`, `renderings/constants.json` |
| The feet above the floor, $`H_{\mathrm{c}}`$, $`W_{20}`$; for (ii), the clear heights in the notches and under the flat parts of the crown | `checks/check_clear_height.py` |
| The section figures | `figures/sections.py` |

### Mobility

| Value | Script |
|---|---|
| One degree of freedom: the null space, also with one hinge locked, in each hall | `checks/check_mobility.py` |
| The lengthening of the rows of blocks (the period-two translation) | `checks/check_mobility.py` |
| The lengthening of $`L`$ | `checks/check_dimensions.py` |
| (iii) at $`t = 1/\sqrt{2}`$: coplanar faces, angles at the other hinges, null space, distance to the samples | `checks/check_mobility.py` |
| The vanishing denominators at $`t = 1/\sqrt{2}`$ | `../gqs.py` (`Block`) |

### Snow check

| Value | Script |
|---|---|
| The shell model and the mesh | `fem/shellnet.py`, `fem/check_results.py` |
| The panel data (assumed values) | `fem/shellnet.py` (`panel_section`), `fem/check_results.py` |
| The snow loads (exposure and thermal coefficients 1, no National Annex), $`h/b`$, $`\mu_3`$ | `fem/shellnet.py`, `fem/check_results.py` |
| The displacement, stress, and buckling limits | `fem/shellnet.py`, `fem/fe.py` |
| The reading radius and the latch zone | `fem/fe.py` (`latch_zone`) |
| The radius against the hinge lines | `checks/check_nets.py` |
| The capacities, $`\lambda_1`$, where each is set, the spread of the foot on the rails, the latch zone of (i) | `fem/check_results.py` |
| The wrinkling capacities of (i) at three radii, the local modes not counted, and the mesh study | `fem/check_results.py` |
| The snow figure | `figures/snow.py` |

### Renderings

| Value | Script |
|---|---|
| The renderings (Blender, Cycles) | `renderings/render.sh` |
| The trailer and the container, with their sources; the schematic parts | `renderings/constants.json`, `renderings/scene.py` |
| The rise, the outer top of the panels, the deployed length, the packed tube of (iii) | `checks/check_dimensions.py` |
| The plinths and knee walls; the gaps under the feet | `checks/check_dimensions.py`, `checks/check_clear_height.py`, `renderings/scene.py` |
| The sub-pixel offsets | `renderings/scene.py` |

## 6. What is not recomputed here

- The FE values come from the stored results of the full runs; only two full static runs were rerun
  (`fem/results/reruns/`).
- No deck was kept of the static runs of (iii) (`fem/fe.py` writes these decks again) or of the separate run behind
  its vertical displacements at $`k = 6`$.
- The searches that found the nets are not included, and their targets are not rechecked.
- The quick run does not recompute the renderings, and a render is not bit-reproducible.
- Whether another motion branches off at $`t = 1/\sqrt{2}`$, and whether the drive with one spindle passes through it,
  were not examined.
