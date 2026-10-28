# Collision check of the printed parts

`motion_sweep.json` holds a numerical collision check of the assembled 1:20 model of the hall (ii): all 187 faces with
their 346 hinges (160 of them curved loop-closing hinges) and all 640 pins pushed home, as designed, with no play. The
check runs over 181 positions of the computed motion, from the deployed position (state 0; t = 0.7265 in the paper, the
position on the stand) to the transport position (state 180; t = 1.4019). Each minimum also gives the angle `th` of the
data at its position (108 at state 0, 71 at state 180).

At every position, the smallest exact signed distance of each kind of pair is found among the convex pieces that come
within 2.5 mm (pairs farther apart are skipped, so minima above 2.5 mm are not exact). A pin is not paired with the
knuckles of its own hinge, which hold it by design. Each kind of pair has a required clearance (`gates`); `minima` gives
the smallest value found over all positions, with the pair and the position, and `per_state` gives the minima at each
position. Result: every requirement holds at every position (`all_gates_passed`: true), and the smallest distance found
is 0.050 mm, so the parts keep their design clearances along the whole motion.

| Pair | Key | Required (mm) | Smallest (mm) |
|---|---|---|---|
| faces and knuckles not joined by a hinge | `nonpartner` | 0.45 | 1.668 |
| a knuckle and a part of another hinge | `ring_other_hinge` | 0.40 | 1.668 |
| the two faces of a hinge | `partner_body_body` | 0.45 | 0.509 |
| a knuckle and the partner face, radially | `partner_ring_body_radial` | 0.40 | 0.400 |
| a knuckle and the partner face, along the axis | `partner_ring_body_axial` | 0.00 | 0.050 |
| the knuckles of a hinge, side gap | `partner_ring_ring` | 0.00 | 0.050 |
| a pin and the parts around it | `pin` | 0.30 | 0.300 |
| curved loop-closing hinges: knuckles, band, partner face radially | `closing_ring_ring`, `closing_ring_band`, `closing_ring_body_radial` | 0.06, 0.06, 0.40 | 0.060, 0.060, 0.400 |
| curved loop-closing hinges: knuckle and partner face along the axis | `closing_ring_body_axial` | 0.06 | no pair within reach (no entry in `minima`) |
| a pin shank and the partner face (information only) | `pin_shank_partner` | | 0.250 |

The smallest values 0.400, 0.300, and 0.060 mm equal their requirements, and the 0.050 mm values are the 0.05 mm side gap
of every knuckle: all are clearances built into the parts, not near misses. The run took 58 min on 8 processes
(`seconds`, `cpu_seconds`). `motion_sweep.log` is the printed summary of the same run.
