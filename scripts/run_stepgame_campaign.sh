#!/bin/bash
# StepGame campaign: Reflexion, Memp and MemoryBank on StepGame for three models.
# Run `baseline` and read `gate` before submitting any arm; drop a model at ceiling
# or chance floor with MODELS.
#
#   bash scripts/run_stepgame_campaign.sh baseline          # 3 jobs
#   bash scripts/run_stepgame_campaign.sh gate              # no GPU; after baseline
#   bash scripts/run_stepgame_campaign.sh calibrate         # 6 jobs: Li et al.'s prompts
#   bash scripts/run_stepgame_campaign.sh calibrate-report  # no GPU; after calibrate
#   bash scripts/run_stepgame_campaign.sh refl-root         # 3 jobs
#   bash scripts/run_stepgame_campaign.sh refl-arms         # 6 jobs; after refl-root
#   bash scripts/run_stepgame_campaign.sh memp-build        # 3 jobs (train pass)
#   bash scripts/run_stepgame_campaign.sh memp-distill      # 3 jobs; after memp-build
#   bash scripts/run_stepgame_campaign.sh memp-embed        # 1 job; needs only the splits
#   bash scripts/run_stepgame_campaign.sh memp-upgrade      # no GPU; after memp-distill
#   bash scripts/run_stepgame_campaign.sh memp-eval         # 9 jobs; after memp-upgrade + memp-embed
#   bash scripts/run_stepgame_campaign.sh mb-distill        # 3 jobs; after memp-build
#   bash scripts/run_stepgame_campaign.sh mb-eval           # 6 jobs; after mb-distill + memp-embed
#   bash scripts/run_stepgame_campaign.sh compare           # no GPU
#
# DRYRUN=1 prints the sbatch lines instead of submitting; use it after any edit, since
# file names must match across stages. MODELS="qwen32b gptoss120b" restricts any stage.
set -euo pipefail
cd "$(dirname "$0")"/..

STAGE="${1:-}"

# The ag_gipp project store is near its inode quota, which breaks load_dataset's
# lock file; keep the dataset cache elsewhere.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/hf_datasets}"
[[ -d "$HF_DATASETS_CACHE" ]] || { echo "no dataset cache at $HF_DATASETS_CACHE" >&2; exit 1; }
ENVX="HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

SPLITS=data/stepgame/splits
# k-stratified split, as in published StepGame results. Named explicitly because
# benchmarks.py's _default() fails when two candidate files exist.
TEST="${TEST:-$SPLITS/stepgame_clean_test_k1-10_n1000.jsonl}"
# Hop-stratified split as the bank source; swapping in the k-stratified file is an ablation.
TRAIN="${TRAIN:-$SPLITS/stepgame_clean_train_byhops_k1-5_n500.jsonl}"
BENCHX="BENCHMARK=stepgame,STEPGAME_TRAIN=$TRAIN,STEPGAME_TEST=$TEST"

# Must match between memp-embed and every eval reading its .npz (the driver checks).
# `lexical` is the retrieval-key ablation.
KEY_VARIANT="${KEY_VARIANT:-structural}"
KEYX="KEY_VARIANT=$KEY_VARIANT"
KTAG=""; [[ "$KEY_VARIANT" == "structural" ]] || KTAG="_$KEY_VARIANT"

# Actor prompt for every stage: Li et al.'s instructions with exemplars stripped,
# so the baseline stays memoryless.
PROMPT_PROTOCOL="${PROMPT_PROTOCOL:-upstream-header}"
PROTOX="PROMPT_PROTOCOL=$PROMPT_PROTOCOL"
# Downstream artefacts carry the protocol tag so runs under different prompts are never paired.
PTAG=""; [[ "$PROMPT_PROTOCOL" == "ours" ]] || PTAG="_$PROMPT_PROTOCOL"

# Retrieval keys do not depend on the actor prompt, so the .npz carries no PTAG.
EMB="results/json/memp_embeddings_stepgame${KTAG}.npz"

# GRANULARITY is in the eval OUT name so granularities never overwrite each other;
# `trajectory` keeps the untagged name.
GRANULARITY="${GRANULARITY:-trajectory}"
case "$GRANULARITY" in
  trajectory)        GTAG="" ;;
  proceduralization) GTAG="_proc" ;;
  *)                 GTAG="_$GRANULARITY" ;;
esac

# tag|model|TP|gpus|extra serve env
ALL_MODELS=(
  "qwen06b|Qwen/Qwen3-0.6B|1|A100:1|"
  "qwen32b|Qwen/Qwen3-32B|2|A100:2|"
  "gptoss120b|openai/gpt-oss-120b|4|A100:4|REASONING_PARSER=openai_gptoss"
)
if [[ -n "${MODELS:-}" ]]; then
  keep=()
  for spec in "${ALL_MODELS[@]}"; do
    for want in $MODELS; do
      [[ "${spec%%|*}" == "$want" ]] && keep+=("$spec")
    done
  done
  [[ ${#keep[@]} -gt 0 ]] || { echo "MODELS='$MODELS' matched none of the three tags" >&2; exit 1; }
  ALL_MODELS=("${keep[@]}")
fi

# Per-stage walltimes, kept tight for backfill. Every phase checkpoints: resume with RESUME=<file>.
t_of() {   # t_of <stage> <tag>
  case "$1:$2" in
    base:qwen06b|root:qwen06b)      echo 02:00:00 ;;
    base:qwen32b|root:qwen32b)      echo 06:00:00 ;;
    base:gptoss120b|root:gptoss120b) echo 04:00:00 ;;
    arm:qwen06b)                    echo 06:00:00 ;;
    arm:*)                          echo 12:00:00 ;;
    build:qwen06b|distill:qwen06b)  echo 02:00:00 ;;
    build:qwen32b|distill:qwen32b)  echo 04:00:00 ;;
    build:*|distill:*)              echo 03:00:00 ;;
    eval:qwen06b)                   echo 03:00:00 ;;
    eval:qwen32b)                   echo 08:00:00 ;;
    eval:*)                         echo 05:00:00 ;;
    *) echo "no walltime for stage '$1' tag '$2'" >&2; exit 1 ;;
  esac
}

sub() {   # sub <jobname> <gpus> <time> <exports...>
  local name=$1 gpus=$2 tlimit=$3; shift 3
  if [[ "${DRYRUN:-0}" == "1" ]]; then
    printf 'sbatch -J %s -G %s -t %s --export=ALL,%s scripts/serve_and_run.sbatch\n' \
           "$name" "$gpus" "$tlimit" "$*"
  else
    sbatch -J "$name" -G "$gpus" -t "$tlimit" \
      --export=ALL,"$*" scripts/serve_and_run.sbatch
  fi
}

case "$STAGE" in
  # ---- baseline gate ----------------------------------------------------------
  baseline)
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      sub "sg-base-$tag" "$gpus" "$(t_of base "$tag")" \
        "$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=baseline,SPLIT=test,LIMIT=0,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=results/json/baseline_stepgame_${tag}_test${PTAG}.json"
    done
    echo "submitted the baseline gate. When it lands:"
    echo "  bash scripts/run_stepgame_campaign.sh gate"
    ;;

  gate)
    # Is there headroom at h>=4, above the chance floor and below the ceiling?
    # A model without it cannot show a memory effect.
    PTAG="$PTAG" python - <<'PY'
import glob, json, pathlib
from collections import Counter
from math import comb

FLOOR = 1 / 9   # nine labels: a lucky-wrong answer is common, unlike SPaRC

def tail(k, n, p):       # exact P(X >= k) under the floor; scipy is not in the venv
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))

import os
PTAG = os.environ.get("PTAG", "")
rows = sorted(f for f in glob.glob(f"results/json/baseline_stepgame_*_test{PTAG}.json")
              if PTAG or "_test." in f)
if not rows:
    raise SystemExit("no baseline_stepgame_*_test.json yet -- has the gate landed?")

# Counted over episodes, not averaged over per-hop rates: the hop buckets are very uneven.
print(f"{'actor':<12} {'overall':>8} {'h<=3':>12} {'h>=4':>12} {'p vs floor':>11}   verdict")
for f in rows:
    d = json.loads(pathlib.Path(f).read_text())
    tag = pathlib.Path(f).stem.split("_")[2]
    n, ok = Counter(), Counter()
    for e in d["episodes"]:
        h = (e.get("extra") or {}).get("hops")
        if h is None:
            continue
        n[h] += 1
        ok[h] += bool(e["success"])
    if not n:
        print(f"{tag:<12}  no per-item hop data"); continue
    lo_n = sum(v for k, v in n.items() if k <= 3); lo_k = sum(v for k, v in ok.items() if k <= 3)
    hi_n = sum(v for k, v in n.items() if k >= 4); hi_k = sum(v for k, v in ok.items() if k >= 4)
    band = hi_k / hi_n
    p = tail(hi_k, hi_n, FLOOR)
    # Two ways to have no headroom: indistinguishable from chance, or at ceiling.
    if p > 0.05:       v = "AT CHANCE FLOOR -> unmeasurable at depth"
    elif band >= 0.95: v = "AT CEILING      -> unmeasurable"
    else:              v = "headroom OK     -> commit the arms"
    print(f"{tag:<12} {d['summary']['accuracy']:>8.3f} "
          f"{lo_k:>5}/{lo_n:<4}={lo_k/lo_n:.2f}" .rjust(0)
          + f" {hi_k:>4}/{hi_n:<4}={band:.2f} {p:>11.3g}   {v}")

print(f"\nfloor = {FLOOR:.3f} (1/9).  per-hop solved/n:")
for f in rows:
    d = json.loads(pathlib.Path(f).read_text())
    n, ok = Counter(), Counter()
    for e in d["episodes"]:
        h = (e.get("extra") or {}).get("hops")
        if h is not None:
            n[h] += 1; ok[h] += bool(e["success"])
    print(f"  {pathlib.Path(f).stem.split('_')[2]:<12} "
          + "  ".join(f"h{k}:{ok[k]}/{n[k]}" for k in sorted(n)))
PY
    ;;

  # ---- calibration against the published protocol ----------------------------
  calibrate)
    # Li et al.'s prompt on the same split and grader; no bearing on the arms.
    # `upstream` is 10-shot and not memoryless: never use it for refl-root, memp-build
    # or a control arm.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      for proto in upstream-header upstream; do
        ptag=hdr; [[ "$proto" == "upstream" ]] && ptag=10sh
        sub "sg-cal-$ptag-$tag" "$gpus" "$(t_of base "$tag")" \
          "$ENVX,$BENCHX,$KEYX,PROMPT_PROTOCOL=$proto,DRIVER=baseline,SPLIT=test,LIMIT=0,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=results/json/baseline_stepgame_${tag}_test_${proto}.json"
      done
    done
    echo "submitted the calibration sweep. When it lands:"
    echo "  bash scripts/run_stepgame_campaign.sh calibrate-report"
    ;;

  calibrate-report)
    # Side by side, per model, per hop-depth band. No GPU.
    python - <<'PY_CAL'
import glob, json, pathlib
from collections import Counter

PROTO = {"": "ours (zero-shot)", "_upstream-header": "upstream-header (decoders)",
         "_upstream": "upstream (decoders + 10 shots)"}

def load(tag, suffix):
    f = pathlib.Path(f"results/json/baseline_stepgame_{tag}_test{suffix}.json")
    return json.loads(f.read_text()) if f.exists() else None

tags = sorted({pathlib.Path(f).stem.split("_")[2]
               for f in glob.glob("results/json/baseline_stepgame_*_test*.json")})
print(f"{'actor':<12} {'protocol':<30} {'overall':>8} {'h<=3':>8} {'h>=4':>8}")
for tag in tags:
    for suffix, label in PROTO.items():
        d = load(tag, suffix)
        if d is None:
            continue
        n, ok = Counter(), Counter()
        for e in d["episodes"]:
            h = (e.get("extra") or {}).get("hops")
            if h is not None:
                n[h] += 1; ok[h] += bool(e["success"])
        lo_n = sum(v for k, v in n.items() if k <= 3) or 1
        hi_n = sum(v for k, v in n.items() if k >= 4) or 1
        lo_k = sum(v for k, v in ok.items() if k <= 3)
        hi_k = sum(v for k, v in ok.items() if k >= 4)
        print(f"{tag:<12} {label:<30} {d['summary']['accuracy']:>8.3f} "
              f"{lo_k/lo_n:>8.3f} {hi_k/hi_n:>8.3f}")
    print()
print("`upstream` is 10-shot: it is a calibration point, NOT a memoryless baseline.")
PY_CAL
    ;;

  # ---- Reflexion (episodic) -------------------------------------------------
  refl-root)
    # Trial 1, shared by both arms. OUT is fixed per model: the auto-name has no model tag.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      sub "sg-root-$tag" "$gpus" "$(t_of root "$tag")" \
        "$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=reflexion,PHASE=root,SPLIT=test,LIMIT=0,TRIALS=10,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=results/json/reflexion_root_stepgame_${tag}_test${PTAG}.json"
    done
    ;;

  refl-arms)
    # The effect is memory minus control at the same trial index: with nine labels,
    # resampling alone raises the curve. DEP_ROOT_<TAG>=<jobid> chains on refl-root.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      root="results/json/reflexion_root_stepgame_${tag}_test${PTAG}.json"
      depvar="DEP_ROOT_$(echo "$tag" | tr '[:lower:]' '[:upper:]')"
      dep="${!depvar:-}"
      [[ -f "$root" || -n "$dep" ]] || {
        echo "skipping $tag: no $root and no $depvar=<jobid>" >&2; continue; }
      for arm in memory control; do
        name="sg-${arm:0:3}-$tag"
        if [[ "${DRYRUN:-0}" == "1" ]]; then
          printf 'sbatch -J %s -G %s -t %s %s--export=ALL,... ARM=%s ROOT=%s\n' \
                 "$name" "$gpus" "$(t_of arm "$tag")" "${dep:+--dependency=afterok:$dep }" "$arm" "$root"
        else
          sbatch -J "$name" -G "$gpus" -t "$(t_of arm "$tag")" \
            ${dep:+--dependency=afterok:$dep} \
            --export=ALL,"$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=reflexion,PHASE=arm,ARM=$arm,ROOT=$root,TRIALS=10,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=results/json/reflexion_${arm}_stepgame_${tag}_test${PTAG}.json" \
            scripts/serve_and_run.sbatch
        fi
      done
    done
    ;;

  # ---- MemP (procedural) ----------------------------------------------------
  memp-build)
    # Memoryless train pass. MemoryBank distils from the same file, so both banks
    # cover identical source items. DEP_BUILD_<TAG>=<jobid> chains later stages on it.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      sub "sg-build-$tag" "$gpus" "$(t_of build "$tag")" \
        "$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=memp,PHASE=build,SPLIT=train,LIMIT=0,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=results/json/memp_build_stepgame_train_${tag}${PTAG}.json"
    done
    ;;

  memp-distill)
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      build="results/json/memp_build_stepgame_train_${tag}${PTAG}.json"
      depvar="DEP_BUILD_$(echo "$tag" | tr '[:lower:]' '[:upper:]')"
      dep="${!depvar:-}"
      [[ -f "$build" || -n "$dep" ]] || {
        echo "skipping $tag: no $build and no $depvar=<jobid>" >&2; continue; }
      if [[ "${DRYRUN:-0}" == "1" ]]; then
        printf 'sbatch -J sg-dist-%s ... BUILD=%s\n' "$tag" "$build"
      else
        sbatch -J "sg-dist-$tag" -G "$gpus" -t "$(t_of distill "$tag")" \
          ${dep:+--dependency=afterok:$dep} \
          --export=ALL,"$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=memp,PHASE=distill,BUILD=$build,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=results/json/memp_bank_stepgame_${tag}${PTAG}.json" \
          scripts/serve_and_run.sbatch
      fi
    done
    ;;

  memp-embed)
    # Serves the embedding model, so it needs its own job. Depends only on the splits.
    sub "sg-emb$KTAG" "A100:1" "01:00:00" \
      "$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=memp,PHASE=embed,TP=1,OUT=$EMB"
    ;;

  memp-upgrade)
    # Runs on the login node, not as a job: it needs no model, and serve_and_run.sbatch
    # has no `upgrade` phase. Writes <bank>_traj.json next to each bank.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag _m _tp _g _e <<< "$spec"
      bank="results/json/memp_bank_stepgame_${tag}${PTAG}.json"
      [[ -f "$bank" ]] || { echo "skipping $tag: no $bank" >&2; continue; }
      echo "=== upgrade $tag ==="
      python scripts/run.py memp --phase upgrade --benchmark stepgame \
        --stepgame-train "$TRAIN" --stepgame-test "$TEST" --bank "$bank"
    done
    ;;

  memp-eval)
    # `fixed` tests whether one item-independent procedure suffices; `random` is the
    # relevance control. Reads the _traj bank written by memp-upgrade.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      bank="results/json/memp_bank_stepgame_${tag}${PTAG}_traj.json"
      [[ -f "$bank" ]] || { echo "skipping $tag: no $bank (run memp-upgrade)" >&2; continue; }
      [[ -f "$EMB" || "${DRYRUN:-0}" == "1" ]] || { echo "skipping $tag: no $EMB (run memp-embed)" >&2; continue; }
      for arm in retrieved random fixed; do
        sub "sg-${arm:0:3}-$tag" "$gpus" "$(t_of eval "$tag")" \
          "$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=memp,PHASE=eval,ARM=$arm,LIMIT=0,GRANULARITY=$GRANULARITY,BANK=$bank,EMBEDDINGS=$EMB,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=results/json/memp_eval_stepgame_${arm}_test_${tag}${KTAG}${PTAG}${GTAG}.json"
      done
    done
    ;;

  # ---- MemoryBank (declarative) --------------------------------------------
  mb-distill)
    # RULES_PREAMBLE=on only; the relation/offset table stands in for SPaRC's rulebook.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      build="results/json/memp_build_stepgame_train_${tag}${PTAG}.json"
      depvar="DEP_BUILD_$(echo "$tag" | tr '[:lower:]' '[:upper:]')"
      dep="${!depvar:-}"
      [[ -f "$build" || -n "$dep" ]] || {
        echo "skipping $tag: no $build and no $depvar=<jobid>" >&2; continue; }
      if [[ "${DRYRUN:-0}" == "1" ]]; then
        printf 'sbatch -J sg-mbd-%s %s... BUILD=%s\n' \
               "$tag" "${dep:+--dependency=afterok:$dep }" "$build"
      else
        sbatch -J "sg-mbd-$tag" -G "$gpus" -t "$(t_of distill "$tag")" \
          ${dep:+--dependency=afterok:$dep} \
          --export=ALL,"$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=memorybank,PHASE=distill,RULES_PREAMBLE=on,BUILD=$build,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=results/json/memorybank_bank_stepgame_${tag}${PTAG}_rp.json" \
          scripts/serve_and_run.sbatch
      fi
    done
    ;;

  mb-eval)
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      bank="results/json/memorybank_bank_stepgame_${tag}${PTAG}_rp.json"
      [[ -f "$bank" ]] || { echo "skipping $tag: no $bank (run mb-distill)" >&2; continue; }
      [[ -f "$EMB" || "${DRYRUN:-0}" == "1" ]] || { echo "skipping $tag: no $EMB (run memp-embed)" >&2; continue; }
      for arm in retrieved random; do
        sub "sg-mb${arm:0:3}-$tag" "$gpus" "$(t_of eval "$tag")" \
          "$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=memorybank,PHASE=eval,ARM=$arm,LIMIT=0,BANK=$bank,EMBEDDINGS=$EMB,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=results/json/memorybank_eval_stepgame_${arm}_test_${tag}${PTAG}_rp.json"
      done
    done
    ;;

  # ---- reports (no GPU) -----------------------------------------------------
  compare)
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag _m _tp _g _e <<< "$spec"
      root="results/json/reflexion_root_stepgame_${tag}_test${PTAG}.json"
      mem="results/json/reflexion_memory_stepgame_${tag}_test${PTAG}.json"
      ctl="results/json/reflexion_control_stepgame_${tag}_test${PTAG}.json"
      if [[ -f "$mem" && -f "$ctl" ]]; then
        echo "=== reflexion $tag ==="
        python scripts/run.py reflexion --phase compare --memory "$mem" --control "$ctl" \
          --out "results/txt/reflexion_vs_control_stepgame_${tag}${PTAG}.txt"
      fi
      # Memp and MemoryBank pair against the Reflexion root (trial 1) where it exists.
      base="$root"; [[ -f "$base" ]] || base="results/json/baseline_stepgame_${tag}_test${PTAG}.json"
      mret="results/json/memp_eval_stepgame_retrieved_test_${tag}${KTAG}.json"
      if [[ -f "$base" && -f "$mret" ]]; then
        echo "=== memp $tag ==="
        python scripts/run.py memp --phase compare --baseline "$base" --retrieved "$mret" \
          $(f="results/json/memp_eval_stepgame_random_test_${tag}${KTAG}${PTAG}.json"; [[ -f "$f" ]] && echo --random "$f") \
          --out "results/txt/memp_vs_baseline_stepgame_${tag}${KTAG}${PTAG}.txt"
      fi
      bret="results/json/memorybank_eval_stepgame_retrieved_test_${tag}_rp.json"
      if [[ -f "$base" && -f "$bret" ]]; then
        echo "=== memorybank $tag ==="
        python scripts/run.py memorybank --phase compare --baseline "$base" --retrieved "$bret" \
          $(f="results/json/memorybank_eval_stepgame_random_test_${tag}${PTAG}_rp.json"; [[ -f "$f" ]] && echo --random "$f") \
          $([[ -f "$mret" ]] && echo --memp "$mret") \
          --out "results/txt/memorybank_vs_baseline_stepgame_${tag}${PTAG}_rp.txt"
      fi
    done
    ;;

  *)
    sed -n '/^#   bash scripts/,/^#   bash scripts.*compare/p' "$0" >&2
    exit 1 ;;
esac
