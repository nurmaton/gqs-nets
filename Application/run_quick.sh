#!/bin/sh
# Runs all fast checks of the three arch halls and compares every measured value with expected_values.json (PASS / FAIL per
# quantity). About 13 min on a laptop; the CalculiX smoke test (about 1-2 min) runs when ccx is found.
#   usage: sh run_quick.sh            environment: PYTHON (default python3), GQS_PY (default ../gqs.py), CCX (default ccx),
#                                     APP_OUT (default ./out), SKIP_FE_SMOKE=1 to leave out the CalculiX run
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

cd "$(dirname "$0")" || exit 2
PY=${PYTHON:-python3}
OUTDIR=${APP_OUT:-$(pwd)/out}
export APP_OUT="$OUTDIR"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run_quick.log"
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }

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
fi
say ""
say ">>> comparison with expected_values.json"
$PY -B checks/compare.py > "$OUTDIR/compare.txt" 2>&1
rc=$?
cat "$OUTDIR/compare.txt" >> "$LOG"
grep -E '^(FAIL|SKIP)' "$OUTDIR/compare.txt" | tee -a "$LOG"
tail -1 "$OUTDIR/compare.txt" | tee -a "$LOG"
say "full table: $OUTDIR/compare.txt; log: $LOG"
exit $rc
