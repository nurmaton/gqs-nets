#!/bin/sh
# Runs all fast checks of the three arch halls and compares every measured value with expected_values.json (PASS / FAIL per
# quantity). About 13 min on a laptop; the CalculiX smoke test (about 1-2 min) runs when ccx is found, the Blender scene
# test (about 30 s) when Blender is found. The result files of an earlier run are removed first, and a step that ends with
# a nonzero exit code makes the run fail.
#   usage: sh run_quick.sh            environment: PYTHON (default python3), GQS_PY (default ../gqs.py), CCX (default ccx),
#                                     BLENDER (default blender), APP_OUT (default ./out), SKIP_FE_SMOKE=1 to leave out the
#                                     CalculiX run, SKIP_RENDER_TEST=1 to leave out the Blender scene test
#   exit code 0 if every step ends with exit code 0 and no quantity FAILs, 1 otherwise, 2 if a requirement is missing
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

cd "$(dirname "$0")" || exit 2
PY=${PYTHON:-python3}
OUTDIR=${APP_OUT:-$(pwd)/out}
export APP_OUT="$OUTDIR"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run_quick.log"
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }
# the result files of an earlier run, with the scene dumps and the ccx files that the scene and smoke tests read, are removed
# so that a step that fails leaves no old values for the comparison; out/skipped.txt lists the optional checks left out
rm -f "$OUTDIR"/check_*.json "$OUTDIR"/make_decks.json "$OUTDIR"/fe_smoke.json "$OUTDIR"/figure_*.json \
      "$OUTDIR"/render_test.json "$OUTDIR"/compare.json "$OUTDIR"/renderings/checks/pre_hall_*
rm -rf "$OUTDIR"/fe/work/hall_ii_V1_k4_static
: > "$OUTDIR/skipped.txt"
FAILED=""

say "Application checks: $(date)"
say "python: $($PY -c 'import sys; print(sys.version.split()[0])' 2>&1)"
if ! $PY -c 'import numpy, scipy, matplotlib' 2>/dev/null; then
    say "numpy, scipy and matplotlib are needed"; exit 2
fi
say "numpy $($PY -c 'import numpy; print(numpy.__version__)'), scipy $($PY -c 'import scipy; print(scipy.__version__)'), matplotlib $($PY -c 'import matplotlib; print(matplotlib.__version__)')"
GQS=${GQS_PY:-../gqs.py}
if [ ! -f "$GQS" ]; then say "gqs.py not found at $GQS (set GQS_PY)"; exit 2; fi
say "gqs.py: $GQS"

step() {
    name=$1; shift
    t0=$(date +%s)
    say ""
    say ">>> $name"
    "$@" >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] || FAILED="$FAILED${FAILED:+; }$name"
    t1=$(date +%s)
    say "    $name: exit code $rc, $((t1 - t0)) s"
}

step "net data (gqs.py rebuild)"            $PY -B -u checks/check_nets.py
step "sizes of the halls"                   $PY -B -u checks/check_dimensions.py
step "clear height"                         $PY -B -u checks/check_clear_height.py
step "mobility and period two"              $PY -B -u checks/check_mobility.py
step "exchange of the types a and c in (iii)" $PY -B -u checks/check_exchange.py
step "panels along the motion (i)"          $PY -B -u checks/check_panels.py i
step "panels along the motion (ii)"         $PY -B -u checks/check_panels.py ii
step "panels along the motion (iii)"        $PY -B -u checks/check_panels.py iii
step "snow check: stored results"           $PY -B -u fem/check_results.py
step "snow check: input decks (md5)"        $PY -B -u fem/make_decks.py --all
CCXBIN=${CCX:-ccx}
if [ "${SKIP_FE_SMOKE:-0}" != "1" ] && command -v "$CCXBIN" >/dev/null 2>&1; then
    export CCX="$CCXBIN"
    step "snow check: CalculiX smoke test, k = 4" $PY -B -u fem/smoke_test.py
else
    say ""; say ">>> CalculiX smoke test skipped (ccx not found or SKIP_FE_SMOKE=1)"
    echo fe_smoke >> "$OUTDIR/skipped.txt"
fi
step "section figures"                      $PY -B -u figures/sections.py
step "snow figure"                          $PY -B -u figures/snow.py
BL=${BLENDER:-blender}
if [ "${SKIP_RENDER_TEST:-0}" != "1" ] && command -v "$BL" >/dev/null 2>&1; then
    export BLENDER="$BL"
    step "renderings: scene build and geometry test" sh renderings/render.sh --test-only
    step "renderings: summary"              $PY -B -u renderings/summary.py
else
    say ""; say ">>> rendering scene test skipped (Blender not found: set BLENDER, or SKIP_RENDER_TEST=1)"
    echo render_test >> "$OUTDIR/skipped.txt"
fi
say ""
say ">>> comparison with expected_values.json"
$PY -B checks/compare.py > "$OUTDIR/compare.txt" 2>&1
rc=$?
cat "$OUTDIR/compare.txt" >> "$LOG"
grep -E '^(FAIL|SKIP)' "$OUTDIR/compare.txt" | tee -a "$LOG"
tail -1 "$OUTDIR/compare.txt" | tee -a "$LOG"
if [ -n "$FAILED" ]; then say "FAIL: steps that ended with a nonzero exit code (messages in the log): $FAILED"; rc=1; fi
say "full table: $OUTDIR/compare.txt; log: $LOG"
exit $rc
