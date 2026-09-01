#!/usr/bin/env bash
# Run the full Chactun comparison matrix: 21 runs, sequentially, one GPU.
#
#   3 arms x 5 folds at seed 0          = 15   the fold spread
#   3 arms x fold 0 at seeds 1 and 2    =  6   the seed spread
#
# Both spreads exist so a difference between arms has to clear both to count.
# Fold 0 seed 0 is shared between them and is run once.
#
# IDEMPOTENT. A run whose output directory already holds model_final.pth is
# skipped, so this can be re-invoked after an interruption without repeating
# ~50 minutes of finished work. To force a rerun, delete that directory.
#
# FAILURE POLICY: keep going. One arm failing at hour 9 must not discard the
# other twenty runs; failures are counted and listed at the end instead.
#
#   ./scripts/run_chactun_matrix.sh              # all 21
#   ./scripts/run_chactun_matrix.sh A B          # only those arms
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --folds-only runs the five folds without the fold-0 seed sweep. The seed
# noise floor was already measured on arms A, B and C (0.80 to 1.39 sd on segm
# AP), and re-measuring it for every later arm costs two runs each to re-derive
# a number that is a property of the training process rather than of the arm.
# Differences are judged against the existing floor; if an arm looks promising,
# its seeds can be run afterwards.
FOLDS_ONLY=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--folds-only" ]; then
    FOLDS_ONLY=1
  else
    ARGS+=("$a")
  fi
done

ARMS=("${ARGS[@]+"${ARGS[@]}"}")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(A B C)
fi

declare -A OUTROOT=(
  [A]="outputs/chactun_A_maskrcnn_default_anchors"
  [B]="outputs/chactun_B_maskrcnn_shifted_anchors"
  [C]="outputs/chactun_C_cascade_shifted_anchors"
  [D]="outputs/chactun_D_maskrcnn_d4_augmentation"
  [E]="outputs/chactun_E_maskrcnn_repeat_sampler"
  [F]="outputs/chactun_F_maskrcnn_hires960"
)

# (arm fold seed) triples, fold spread first so the primary result lands early
JOBS=()
for arm in "${ARMS[@]}"; do
  for fold in 0 1 2 3 4; do
    JOBS+=("$arm $fold 0")
  done
done
if [ "$FOLDS_ONLY" -eq 0 ]; then
  for arm in "${ARMS[@]}"; do
    for seed in 1 2; do
      JOBS+=("$arm 0 $seed")
    done
  done
fi

LOGDIR="${REPO}/outputs/matrix_logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY="${LOGDIR}/summary-${STAMP}.txt"

echo "Chactun matrix: ${#JOBS[@]} runs, arms ${ARMS[*]}" | tee "$SUMMARY"
echo "started $(date -Is)" | tee -a "$SUMMARY"
echo | tee -a "$SUMMARY"

n=0
skipped=0
failed=()
START_ALL=$(date +%s)

for job in "${JOBS[@]}"; do
  read -r arm fold seed <<<"$job"
  n=$((n + 1))
  outdir="${OUTROOT[$arm]}/fold${fold}_seed${seed}"
  tag="arm${arm}-fold${fold}-seed${seed}"

  if [ -f "${outdir}/model_final.pth" ]; then
    echo "[${n}/${#JOBS[@]}] ${tag}: SKIP (already complete)" | tee -a "$SUMMARY"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[${n}/${#JOBS[@]}] ${tag}: starting $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  START=$(date +%s)

  ./scripts/run.sh python scripts/train_chactun.py \
      --arm "$arm" --fold "$fold" --seed "$seed" \
      >"${LOGDIR}/${tag}.log" 2>&1
  rc=$?

  MINS=$(( ($(date +%s) - START) / 60 ))
  if [ $rc -ne 0 ]; then
    echo "[${n}/${#JOBS[@]}] ${tag}: FAILED rc=${rc} after ${MINS}m -- see ${LOGDIR}/${tag}.log" \
      | tee -a "$SUMMARY"
    failed+=("$tag")
    continue
  fi

  # pull the headline numbers out of the run's own metrics rather than reparsing
  # stdout, which is noisy and format-unstable
  line=$(python3 - "$outdir" <<'PY'
import json, sys, os
p = os.path.join(sys.argv[1], "metrics.json")
try:
    rows = [json.loads(l) for l in open(p)]
except Exception as e:
    print("metrics unreadable: %s" % e); raise SystemExit
ev = [r for r in rows if "segm/AP" in r]
if not ev:
    print("no evaluation rows"); raise SystemExit
r = ev[-1]
print("segmAP %.2f  AP50 %.2f  AP75 %.2f  bld %.2f  plat %.2f  agu %.2f" % (
    r["segm/AP"], r.get("segm/AP50", float("nan")), r.get("segm/AP75", float("nan")),
    r.get("segm/AP-building", float("nan")), r.get("segm/AP-platform", float("nan")),
    r.get("segm/AP-aguada", float("nan"))))
PY
)
  echo "[${n}/${#JOBS[@]}] ${tag}: done ${MINS}m  ${line}" | tee -a "$SUMMARY"
done

HOURS=$(( ($(date +%s) - START_ALL) / 3600 ))
MINS_T=$(( (($(date +%s) - START_ALL) % 3600) / 60 ))
echo | tee -a "$SUMMARY"
echo "finished $(date -Is) after ${HOURS}h${MINS_T}m" | tee -a "$SUMMARY"
echo "skipped ${skipped}, failed ${#failed[@]}" | tee -a "$SUMMARY"
if [ ${#failed[@]} -gt 0 ]; then
  printf 'FAILED: %s\n' "${failed[*]}" | tee -a "$SUMMARY"
fi
echo "summary: ${SUMMARY}"
