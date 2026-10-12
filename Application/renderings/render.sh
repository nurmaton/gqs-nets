#!/bin/sh
# Renders the three halls (transport and deployed position in one image) with Blender (Cycles) into out/renderings/.
# Per image: (1) build the scene without rendering and dump the evaluated meshes and the camera; (2) run the geometry test on
# that dump (render only if it ends in ': PASS'); (3) render with the settings of cameras.json (1024 samples, 3000 x 1200,
# seed 7; the next seed of 11, 19, 23 if the artefact scan finds denoiser artefacts); (4) test the dump of the render run.
# The scene build and test take seconds; a render takes tens of minutes. Renders are not bit-reproducible (denoiser, Blender
# version); the geometry test is the reproducible part.
#   usage: sh renderings/render.sh [--test-only]       environment: BLENDER (default: blender on the PATH), PYTHON, APP_OUT
#   requires Blender 5.x and Python 3 with numpy, scipy, pillow
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

cd "$(dirname "$0")/.." || exit 2
BL=${BLENDER:-blender}; PY=${PYTHON:-python3}
OUTDIR=${APP_OUT:-$(pwd)/out}; export APP_OUT="$OUTDIR"
G="$OUTDIR/renderings/geo"; C="$OUTDIR/renderings/checks"; R="$OUTDIR/renderings"
mkdir -p "$C"
$PY -B renderings/make_geo.py || exit 1
status=0
for h in i ii iii; do
    if [ $h = i ]; then ARGS="--size 3000x1200 --az 22 --el 26 --gap 6.0"; else ARGS="--size 3000x1200 --az 32 --el 22 --gap 5.0"; fi
    $BL -b --python renderings/scene.py -- "$G/hall_$h.json" "$C/pre_hall_${h}_two.png" $ARGS --dump "$C/pre_hall_${h}_two_scene.npz" \
        --dump-only --camera "$C/camera_hall_$h.json" > "$C/pre_hall_${h}_scene.log" 2>&1
    $PY -B renderings/test_scene.py "$C/pre_hall_${h}_two_scene.npz" "$C/pre_hall_${h}_two_test.json" > "$C/pre_hall_${h}_test.log" 2>&1
    tail -1 "$C/pre_hall_${h}_test.log"
    if ! tail -1 "$C/pre_hall_${h}_test.log" | grep -q ": PASS$"; then echo "hall_$h: geometry test did not pass, not rendered"; status=1; continue; fi
    [ "$1" = "--test-only" ] && continue
    for seed in 7 11 19 23; do
        $BL -b --python renderings/scene.py -- "$G/hall_$h.json" "$R/hall_${h}_two.png" $ARGS --lines --line 1.0 --samples 1024 --view Standard \
            --denoise 1 --adaptive 0.003 --seed $seed --dump "$C/hall_${h}_two_scene.npz" --mask "$R/hall_${h}_two_mask.png" > "$C/hall_${h}_render.log" 2>&1
        $PY -B renderings/artifact_scan.py "$R/hall_${h}_two.png" > "$C/hall_${h}_artifacts.log" 2>&1
        if grep -q "streak tiles 0 \[\], blotch tiles 0" "$C/hall_${h}_artifacts.log"; then echo "hall_$h: rendered, clean with seed $seed"; break; fi
        echo "hall_$h: artefact with seed $seed"
    done
    $PY -B renderings/test_scene.py "$C/hall_${h}_two_scene.npz" "$C/hall_${h}_two_test.json" > "$C/hall_${h}_test.log" 2>&1
    tail -1 "$C/hall_${h}_test.log"
done
exit $status
