# Checks: a collision check of the printed parts

This folder holds the result, `motion_sweep.json`, and its summary, `motion_sweep.log`.

The check covers all faces, hinges, and pins of the 1:20 physical model, with the pins pushed home and no play, at 181
positions from the deployed position (state 0, on the stand) to the transport position (state 180). At each position, the smallest exact signed distance of each kind of pair is found among the convex pieces that come within 2.5 mm. A pin is not paired
with its own knuckles.

All requirements hold (`all_gates_passed`: true):

| Pair | Key | Required (mm) | Smallest (mm) |
|---|---|---|---|
| Faces and knuckles not joined by a hinge | `nonpartner` | 0.45 | 1.668 |
| Knuckle and a face hinged to its face by another hinge | `ring_other_hinge` | 0.40 | 1.668 |
| Faces of a hinge | `partner_body_body` | 0.45 | 0.509 |
| Knuckle and partner face, radially | `partner_ring_body_radial` | 0.40 | 0.400 |
| Knuckle and partner face, axially | `partner_ring_body_axial` | 0.00 | 0.050 |
| Knuckles of a hinge, side gap | `partner_ring_ring` | 0.00 | 0.050 |
| Pin and the parts around it | `pin` | 0.30 | 0.300 |
| Curved loop-closing hinges: knuckles, band, partner face radially | `closing_ring_ring`, `closing_ring_band`, `closing_ring_body_radial` | 0.06, 0.06, 0.40 | 0.060, 0.060, 0.400 |
| Curved loop-closing hinges: knuckle and partner face, axially | `closing_ring_body_axial` | 0.06 | No pair within 2.5 mm |
| Pin shank and partner face (information only) | `pin_shank_partner` | None | 0.250 |

Curved loop-closing hinges: at each of the 160 inner vertices, the hinge that closes the loop of four faces around it, with curved knuckles centred on that vertex. Band: the strip of a face along its knuckles.

Values equal to their requirement and the 0.050 mm side gaps are design clearances.
