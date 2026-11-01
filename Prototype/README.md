# Prototype: a 1:20 physical model of the arch hall (ii)

This folder holds the print files, the assembly guide, and a collision check of the model; the data of the hall are in
[`Application/`](../Application/README.md). The model shows the motion of the net qualitatively, without the knee walls
and the drive of the hall.

## Contents

| File | Content |
|---|---|
| `assembly_guide.html` | The 3D assembly guide |
| `print_files/` | 17 print files for the Bambu Lab H2C |
| [`checks/`](checks/README.md) | The collision check of the printed parts along the motion |

## Printing

The files are sliced for the left nozzle (0.4 mm) of the H2C in white PLA, black for the stand, on the Textured PEI
plate at 55 °C. The fits of bores and pins rely on the slicer settings: a 0.20 mm first layer, then 0.12 mm layers,
3 walls (6 for the stand), 100 % infill, and no elephant-foot compensation. For another printer, re-slice the meshes in
each file (`3D/Objects/`) with these settings; the stand is 299 × 264 mm, and its title is a negative part.

First, print `WHITE_FIT_TEST_PLA_H2C.gcode.3mf` (tagged black; map it to the white PLA). Keep each half-pin with its
tag, and check by hand that:

1. a half-pin of tag 2 goes into the bore A5 and stays when the bar is shaken upside down;
2. H5 and the part 5-05, joined by a snap pin, swing freely and do not slide along the pin;
3. H5 and the part 5-05, joined by the two half-pins of tag 2, one from each end, swing freely, and the pins stay in.

If a check fails, do not print the hall; other sizes need new meshes.

Then print the files in the table's order, because a face takes its half-pins as soon as it comes off the plate. If a
pin comes loose in the first layers, skip its object; the pin plates hold spares. No file uses a brim: watch the first
hour of the stand and the first 30 minutes of the person, and stop if a corner or an end lifts.

| File | Content |
|---|---|
| `WHITE_FIT_TEST_PLA_H2C.gcode.3mf` | Fit test |
| `P01_PINS_ROWS00-03.gcode.3mf` | Pins for rows 0 to 3 |
| `P02_ROW00.gcode.3mf` to `P05_ROW03.gcode.3mf` | Rows 0 to 3 |
| `P06_PINS_ROWS04-07.gcode.3mf` | Pins for rows 4 to 7 |
| `P07_ROW04.gcode.3mf` to `P10_ROW07.gcode.3mf` | Rows 4 to 7 |
| `P11_PINS_ROWS08-10.gcode.3mf` | Pins for rows 8 to 10 |
| `P12_ROW08.gcode.3mf` to `P14_ROW10.gcode.3mf` | Rows 8 to 10 |
| `STAND_FRAME.gcode.3mf` | Stand |
| `PERSON_1TO20_B_ARMWEB.gcode.3mf` | A 1:20 person: two halves and 27 small pins |

By the slicer's estimates, all 17 files take 39 h 36 min, 531.3 g of white PLA, and 147.2 g of black PLA; the hall
alone, 33 h 46 min and 511.2 g.

## Assembly

The [assembly guide](https://nurmaton.github.io/gqs-nets/Prototype/assembly_guide.html) needs a browser with WebGL
(Safari 16.4, Chrome or Edge 87, Firefox 113, or newer) and an internet connection for three.js. Build the hall row by
row, from row 0 to row 10.

| Pin | Use |
|---|---|
| Snap pin: long, hexagonal head | One per hinge with an end on the boundary, pushed in from that end until it clicks |
| Half-pin: short, round head | Two per other hinge, one from each end, put into the face before joining and pushed home after |

Tools: tweezers, and a 2 mm rod for the pins of the 15 long hinges, which sit about 19 mm deep. Carry the hall, also
when part-built, by its feet or flat on a board, never by one face.

## The stand

1. Put the stand on a flat table.
2. Hold the hall by its feet, row 0 towards the stand's title, and lower it straight down into the foot channels without
   pressing.
3. Lift the hall straight up to take it off; fold and open it only off the stand.
4. Carry the loaded stand with both hands under its long sides.

## The person

1. Keep the three rows of pins apart; they differ by 0.1 mm.
2. Flex the plate to free each half whole, lift it by the body, and trim any first-layer lip off its cut face.
3. Try a long pin of row 1, nearest the figure, flat side towards the feet, in a torso hole of the back half; if it
   needs hard pressure, try row 2, then row 3.
4. Push the 5 long and 2 short pins of that row into the back half, the short ones into the lower legs, and put the front half on it.
5. Press the halves together with a flat block, never on the arms; if a gap stays, sand about 0.1 mm off the pin ends.

Hold the person by the torso.
