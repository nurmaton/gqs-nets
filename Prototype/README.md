# Prototype: a 1:20 physical model of the arch hall (ii)

This folder holds the print files, the assembly guide, and a collision check of a 1:20 physical model of the arch hall (ii) of the paper; the
data of the hall are in [`../Application/`](../Application/README.md). The 187 faces of the model are 4 mm thick, that is,
80 mm at full size, and extend to both sides of the net, 2.10 mm outwards and 1.90 mm inwards; the axes of its 346 hinges
lie on the edges of the net, and 640 printed pins join the faces. On its stand, the net is 252.7 mm wide, 257.3 mm long, and
140.1 mm high. Off the stand, the model folds by hand into a nearly closed loop like that of the transport position and
opens again. Its hinges turn freely and have little play: by design, pins of 2.8 mm diameter turn in bores 0.05 mm wider,
with axial gaps of 0.05 to 0.06 mm. Along the computed motion from the deployed to the transport position, the printed
parts were checked numerically at 181 positions: they keep their design clearances, and faces not joined by a hinge stay
at least 1.6 mm apart ([`checks/`](checks/README.md)). The model shows the motion of the net qualitatively, without the knee walls and the
drive of the hall.

Made with the assistance of the Claude model Opus 5.5 (Anthropic); all files were checked by the authors.

## Contents

| File | Content |
|---|---|
| `README.md` | This file: printing, assembly, and the stand. |
| `assembly_guide.html` | The 3D assembly guide: every step, one tab per row of the hall; see [Assembly](#assembly). |
| `print_files/` | 17 print files (`.gcode.3mf`) for the Bambu Lab H2C: a fit test, the 14 plates of the hall (11 of faces, 3 of pins), the stand, and a 1:20 person. |
| `checks/` | The collision check of the printed parts along the motion: result (`motion_sweep.json`), summary (`motion_sweep.log`), and its README. |

## Printing

The files are sliced for the Bambu Lab H2C (left nozzle, 0.4 mm) in PLA, on the Textured PEI plate at 55 °C. Each file
prints one plate with one filament: white PLA for the hall, its pins, the person, and the fit test, and black PLA for the
stand. Map the filament to a slot with that colour when the print starts; the fit test is tagged black, but it is meant
for the white PLA of the hall. Each file also contains the meshes it was sliced from (`3D/Objects/` inside the archive).
For another printer, slice them again with the settings stored in each file, because the fits of the bores and pins rely
on them: a 0.20 mm first layer, then 0.12 mm layers, 3 walls (6 for the stand), 100 % infill, and no elephant-foot
compensation. The title of the stand is a negative part, cut into the frame by the slicer.

First, print `WHITE_FIT_TEST_PLA_H2C.gcode.3mf`. It holds two bars with test bores (A5 to A20 and the edge H28; b5 to b15
and the edge H5), the parts marked 5-05 and 5-10, half-pins beside the tags 1, 2, and 4, and two snap pins. H5 with the
part 5-05 is a hinge with the sizes of the hall, and the half-pins of tag 2 are those of the hall. Keep each half-pin with
its tag when you take them off. Check by hand:

1. a half-pin of tag 2 goes into the bore A5 and stays when the bar is shaken upside down;
2. H5 and the part 5-05, joined by a snap pin, swing freely and do not slide along the pin;
3. H5 and the part 5-05, joined by the two half-pins of tag 2, one from each end, swing freely, and the pins stay in.

If a check fails, do not print the hall: its bores and pins have the sizes of this test, and other sizes need new meshes.

Then print the hall in the order of the table. Each pin plate comes before the rows it serves, because a face takes its
half-pins as soon as it comes off the plate. If a pin comes loose in the first layers, skip its object on the printer; the
pin plates hold spares. No file uses a brim: watch the first hour of the stand and the first 30 minutes of the person, and
stop the print if a corner or an end lifts.

| File | Content | Time | Filament |
|---|---|---|---|
| `WHITE_FIT_TEST_PLA_H2C.gcode.3mf` | fit test | 25 min | 3.58 g |
| `P01_PINS_ROWS00-03.gcode.3mf` | pins for rows 0-3: 238 half-pins, 27 snap pins | 1 h 40 min | 14.70 g |
| `P02_ROW00.gcode.3mf` | row 0: the 17 faces 0-0 to 0-16 | 2 h 33 min | 45.78 g |
| `P03_ROW01.gcode.3mf` | row 1: 1-0 to 1-16 | 2 h 40 min | 42.24 g |
| `P04_ROW02.gcode.3mf` | row 2 | 2 h 40 min | 42.25 g |
| `P05_ROW03.gcode.3mf` | row 3 | 2 h 41 min | 42.30 g |
| `P06_PINS_ROWS04-07.gcode.3mf` | pins for rows 4-7: 273 half-pins, 10 snap pins | 1 h 44 min | 14.84 g |
| `P07_ROW04.gcode.3mf` | row 4 | 2 h 41 min | 42.29 g |
| `P08_ROW05.gcode.3mf` | row 5 | 2 h 42 min | 42.33 g |
| `P09_ROW06.gcode.3mf` | row 6 | 2 h 42 min | 42.35 g |
| `P10_ROW07.gcode.3mf` | row 7 | 2 h 42 min | 42.39 g |
| `P11_PINS_ROWS08-10.gcode.3mf` | pins for rows 8-10: 137 half-pins, 22 snap pins | 1 h 03 min | 9.09 g |
| `P12_ROW08.gcode.3mf` | row 8 | 2 h 42 min | 42.39 g |
| `P13_ROW09.gcode.3mf` | row 9 | 2 h 42 min | 42.40 g |
| `P14_ROW10.gcode.3mf` | row 10 | 2 h 27 min | 45.83 g |
| `STAND_FRAME.gcode.3mf` | the stand (black) | 4 h 20 min | 147.2 g |
| `PERSON_1TO20_B_ARMWEB.gcode.3mf` | the 1.80 m person of the paper's renderings at 1:20 (90 mm tall), with a web between each arm and the torso: two halves and 27 small pins | 1 h 04 min | 16.59 g |

The times are the slicer's estimates and the filament its net extrusion; the start-up purge and prime line add a little
more. The 14 plates of the hall take 33 h 46 min and 511.2 g; all 17 files take 39 h 36 min, 531.3 g of white PLA, and
147.2 g of black PLA (totals from the unrounded slicer values).

## Assembly

The guide shows every step of the hall in 3D: open <https://nurmaton.github.io/gqs-nets/Prototype/assembly_guide.html> in a
current browser with WebGL (Safari 16.4, Chrome or Edge 87, Firefox 113, or newer) and an internet connection (the page
loads three.js).

Each face carries its name i-j (row i from 0 to 10, column j from 0 to 16; face i-j is F_(i+1)(j+1) of the paper) and its
hinge numbers on the numbered side, the inside of the hall. Two pin kinds join the faces; keep them apart:

- **snap pin**, long (15.5 mm) with a hexagonal head: one in each of the 52 hinges with an end on the boundary of the hall,
  pushed in from that end until it clicks;
- **half-pin**, short (6.85 mm) with a round head: two in each of the other 294 hinges, one from each end (588 in all), put
  into the face before it is joined, their tips flush with the inner face of the knuckle or about 0.3 mm inside the hole,
  as the guide shows, and pushed home after the hinge is joined.

The pin plates hold 59 snap pins and 648 half-pins, spares included. Tools: tweezers, and a 2 mm rod (for example, a drill
shank) for the pins of the 15 long hinges, which sit about 19 mm deep.

Build the hall row by row, from row 0 to row 10, as the guide shows; each row is one arch from one foot over the ridge to the
other. Ten corner faces, in columns 1 and 5 of rows 2, 4, 6, 8, and 10, have no straight way in; the guide shows two ways
to put them on by hand. Carry the hall, also when part-built, by its feet or flat on a board, never by one face.

The person: take its pins off the plate one row at a time and keep the three rows apart; their pins differ by 0.1 mm from
row to row. Flex the plate until each half comes loose whole, arms included; lift it by the body, never by an arm, and trim
any lip of the first layer off its cut face. Pins and holes are D-shaped: push one long pin of the row nearest the figure
(row 1), flat side towards the feet, into a torso hole of the back half. If it needs hard pressure, pull it out with
tweezers and try row 2, then row 3. Push the 5 long and the 2 short pins of that row into the back half until they stop,
lay it cut face up, put the front half on it, and press the halves together with a flat block, in turn over the legs,
pelvis, torso, and head, never on the arms. If a gap stays between the cut faces, open the halves, remove any stray thread
or lip, and sand about 0.1 mm off the pin ends. Hold the person by the torso.

## The stand

The stand is an open frame of 264 × 299 mm with one channel for each foot of the hall and the title on its front bar.

1. Put the stand on a flat table.
2. Hold the hall by its feet, row 0 towards the title, and lower it straight down: every foot corner goes into a channel, and
   the corners at the end of row 10 drop into the two pockets at the back. Do not press it in; it sits by its own weight.
3. To take the hall off, lift it straight up out of the channels. Fold and open it only off the stand, never in the channels.
4. With the hall on it, carry the stand with two hands under its long sides, never by one end bar or by a corner.

## License

The files of this folder are under the MIT license of the repository; see [`LICENSE`](../LICENSE).
