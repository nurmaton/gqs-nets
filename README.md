# Generalized quasi-symmetric nets (GQS-nets): notebooks, code, and data

For the paper *Generalized Quasi-Symmetric Nets: A Constructive Approach to the Equimodular
Elliptic Type and to Flexible m × n Nets*: Mathematica notebooks that verify the symbolic computations, a Python tool
that builds and moves the flexible m × n nets, and the data and checks of three transportable arch halls.

## Video

*Coming soon.*

<!-- Placeholder for the video: replace the line above, for example by a thumbnail linked to the video:
[![Video](video-thumbnail.png)](https://VIDEO-LINK) -->

## Contents

| File | Content |
|---|---|
| `helper.nb` | Mathematica notebook: verifies the symbolic computations of the proofs; the paper cites each section where it is used. |
| `ehelper.nb` | Mathematica notebook: verifies the computations behind the construction of the m × n nets and their examples; the paper cites each section where it is used. |
| `gqs.py` | Python tool: assembles a flexible m × n net from 3 × 3 GQS-nets, shows its flexion with a slider, double-checks the proved properties numerically, tests the net for self-intersections, and exports pictures (PNG) and meshes (OBJ). |
| `parameters.txt` | The settings of `gqs.py` for the nets shown in the paper (the digits 2026, the letters G, Q, S, and the animals) and how to reproduce each picture. |
| `Application/` | Three transportable arch halls built from these nets: net data, checks of sizes, clear height, panels, mobility, and snow load (CalculiX), figures, and renderings; see [`Application/README.md`](Application/README.md). |
| `Prototype/` | A 1:20 physical model of the arch hall (ii) that folds by hand: print files for the Bambu Lab H2C, a 3D assembly guide, and a collision check of the printed parts along the motion; see [`Prototype/README.md`](Prototype/README.md). |

## Requirements and use

- **Notebooks:** written with Mathematica 14.0; open a notebook and evaluate it.
- **gqs.py:** Python 3 with numpy 1.20 or newer and matplotlib 3.7 or newer.
  ```
  python3 gqs.py
  ```
  opens the interactive window; `python3 gqs.py --frames` saves six frames of the flexion without a window, and
  `python3 gqs.py --help` lists the options.
- **Application:** Python 3 with numpy, scipy, and matplotlib; CalculiX and Blender for some of the checks.
  ```
  sh Application/run_quick.sh
  ```
  runs all fast checks, writes PASS, FAIL, or SKIP for every checked quantity to `Application/out/compare.txt`, and
  prints the failed and skipped quantities and the totals; versions, run times, and the longer runs are described in
  [`Application/README.md`](Application/README.md).

## Note

The code was written or revised with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic);
all code was checked by the authors.

## Citation

*To be added after publication.*
