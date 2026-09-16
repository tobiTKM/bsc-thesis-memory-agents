#!/bin/bash
# The StepGame campaign: all three memory methods over StepGame, three models.
#
# Mirrors run_memorybank_campaign.sh / rerun_memp_v2.sh because it mirrors their
# experiment -- same three actors, same seed and sampling, same paired
# exact-McNemar contrasts -- on a different task. See docs/stepgame.md.
#
# WHY STAGED, AND WHY YOU SHOULD NOT JUST RUN THEM ALL:
# docs/stepgame.md section 6 risk 1 is a real risk, not a formality. gpt-oss-120b may
# already be at ceiling at low hop counts and Qwen3-0.6B at the ~11% chance floor,
# and memory is unmeasurable in either. Run `baseline`, READ THE by_hops COLUMN,
# and only then commit GPU to the arms -- dropping a model here saves ~20 jobs.
#
#   bash scripts/run_stepgame_campaign.sh baseline       # THE GATE: 3 jobs, ~20-90 min
#   bash scripts/run_stepgame_campaign.sh gate           # read the gate (no GPU)
#   bash scripts/run_stepgame_campaign.sh calibrate      # 6 jobs: Li et al.'s prompt
#   bash scripts/run_stepgame_campaign.sh refl-root      # 3 jobs
#   bash scripts/run_stepgame_campaign.sh refl-arms      # 6 jobs (memory + control)
#   bash scripts/run_stepgame_campaign.sh memp-build     # 3 jobs (train pass)
#   bash scripts/run_stepgame_campaign.sh memp-distill   # 3 jobs
#   bash scripts/run_stepgame_campaign.sh memp-embed     # 1 job  (1 GPU, ~1 h)
#   bash scripts/run_stepgame_campaign.sh memp-upgrade   # NO GPU, login node
#   bash scripts/run_stepgame_campaign.sh memp-eval      # 9 jobs (retrieved/random/fixed)
#   bash scripts/run_stepgame_campaign.sh mb-distill     # 3 jobs
#   bash scripts/run_stepgame_campaign.sh mb-eval        # 6 jobs
#   bash scripts/run_stepgame_campaign.sh compare        # NO GPU
#
# DRYRUN=1 prints the submit lines instead of submitting them. Use it after any
# edit here: every line below names files that must match across stages, and a
# typo in an EMBEDDINGS or BANK path stays invisible until the numbers disagree.
# MODELS="qwen32b gptoss120b" restricts any stage to a subset (what the gate is for).
set -euo pipefail
cd "$(dirname "$0")"/..

STAGE="${1:-}"

# The ag_gipp PROJECT store sits near its 7M-inode quota, which makes load_dataset's
# lock file creation fail (see rerun_memp_v2.sh for the full diagnosis). StepGame
# reads frozen JSONL rather than a HF dataset, so this is belt-and-braces here --
# but the drivers import `datasets` regardless and a stage that touches SPaRC would
# hit it, so it stays set for consistency with the other two campaigns.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/vast-standard/home/t.mark/u21167/hf_datasets}"
[[ -d "$HF_DATASETS_CACHE" ]] || { echo "no dataset cache at $HF_DATASETS_CACHE" >&2; exit 1; }
ENVX="HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

SPLITS=data/stepgame/splits
# TEST is the k-stratified split: "the axis every published StepGame number uses;
# our results must be comparable to them" (docs/stepgame.md 2.2). It is also
# benchmarks.py's default, but named explicitly here because _default() treats two
# candidate files as a fatal ambiguity and every arm must score the IDENTICAL items.
TEST="${TEST:-$SPLITS/stepgame_clean_test_k1-10_n1000.jsonl}"
# TRAIN is the HOP-stratified split, i.e. the bank source. docs/stepgame.md 4.3 gives
# three arguments for it over the k-stratified one: 465/500 distinct structural keys
# against 370 (largest tied bucket 6 against 19), and the k-stratified train split is
# 58% one-hop -- a bank of worked examples of a task with no chaining in it. Swapping
# this line to the k-stratified file is the documented ablation.
TRAIN="${TRAIN:-$SPLITS/stepgame_clean_train_byhops_k1-5_n500.jsonl}"
BENCHX="BENCHMARK=stepgame,STEPGAME_TRAIN=$TRAIN,STEPGAME_TEST=$TEST"

# Must match between the embed job and every eval that reads its .npz -- the driver
# refuses the pair if it does not. `lexical` is the retrieval-key ablation (4.3).
KEY_VARIANT="${KEY_VARIANT:-structural}"
KEYX="KEY_VARIANT=$KEY_VARIANT"
KTAG=""; [[ "$KEY_VARIANT" == "structural" ]] || KTAG="_$KEY_VARIANT"

# The actor prompt every stage runs under. Decision 2026-09-13: `upstream-header`
# -- Li et al.'s instruction paragraph (clock + cardinal decoders), exemplars
# stripped so the baseline stays memoryless. It scores LOWER than `ours`
# (Qwen3-32B 0.690 vs 0.828; 0.449 vs 0.750 at h>=4), which is the point: it
# leaves headroom in the informative band. See docs/stepgame.md 3.2 for the
# measured trade-off, INCLUDING the stated confound -- roughly half the gap is
# frame inversion, and >50% of the distilled bank entries already state the
# reference-frame convention, so part of any memory gain is memory re-supplying
# what this prompt omits. That must be reported alongside the effect size.
PROMPT_PROTOCOL="${PROMPT_PROTOCOL:-upstream-header}"
PROTOX="PROMPT_PROTOCOL=$PROMPT_PROTOCOL"
# Every artefact downstream of the actor prompt is tagged with it, so a run under
# one protocol can never be paired against a run under another -- which the McNemar
# tests could not detect, since both files are internally consistent.
PTAG=""; [[ "$PROMPT_PROTOCOL" == "ours" ]] || PTAG="_$PROMPT_PROTOCOL"

# Embeddings are keyed on retrieval-key text, which does not depend on the actor
# prompt -- so the .npz is shared across protocols and carries no PTAG.
EMB="results/json/memp_embeddings_stepgame${KTAG}.npz"

# GRANULARITY is part of the eval OUT name, because it was NOT: a second granularity
# over the same bank would silently overwrite the first, and both files are
# internally consistent so nothing downstream could catch it. Empty tag for
# `trajectory` so the names of the runs already on disk are preserved.
GRANULARITY="${GRANULARITY:-trajectory}"
# Short tag, so `proceduralization` does not make an already-long filename worse.
case "$GRANULARITY" in
  trajectory)        GTAG="" ;;
  proceduralization) GTAG="_proc" ;;
  *)                 GTAG="_$GRANULARITY" ;;
esac

# tag|model|TP|gpus|extra serve env
# Per-model knobs are rerun_memp_v2.sh's, verbatim: gpt-oss needs its own reasoning
# parser, and TP is sized to the checkpoint. Time limits are per stage, below --
# StepGame is ONE call per item against SPaRC's ~15 steps per puzzle, so these are
# a fraction of the SPaRC campaign's, with headroom for the model load off ceph.
ALL_MODELS=(
  "qwen06b|Qwen/Qwen3-0.6B|1|A100:1|"
  "qwen32b|Qwen/Qwen3-32B|2|A100:2|"
  "gptoss120b|openai/gpt-oss-120b|4|A100:4|REASONING_PARSER=openai_gptoss"
)
# MODELS="qwen32b gptoss120b" drops a model the gate showed to be unmeasurable.
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

# Per-stage walltimes, keyed by tag. Kept as tight as the observed SPaRC runtimes
# allow: on a saturated partition an oversized limit blocks backfill, and every
# phase checkpoints, so an overrun resumes with RESUME=<file>.
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
  # ---- THE GATE -------------------------------------------------------------
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
    # The decision docs/stepgame.md section 6 risk 1 asks for, made explicit: is there
    # measurable headroom in the informative band (h>=4), or is this actor pinned at
    # the ceiling or the ~11% chance floor? A model with no headroom cannot show a
    # memory effect in either direction, and its ~20 remaining jobs are wasted.
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

# Counted over EPISODES, not averaged over summary.accuracy_by_hops. On the
# k-stratified split the hop buckets are wildly uneven (406 items at h=1 against 5
# at h=9 and 4 at h=10), so a mean of per-bucket rates weights four items as
# heavily as four hundred. It reported h>=4 = 0.169 for Qwen3-0.6B where the real
# item-weighted figure is 0.127 -- the difference between "headroom" and "floor".
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
    # Two ways to have no headroom, for opposite reasons. The floor test is the
    # honest one: 12.7% *looks* like signal until you notice chance is 11.1%.
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

  # ---- calibration against the published protocol (no bearing on the arms) ---
  calibrate)
    # Our actor prompt is zero-shot by design: few-shot exemplars are themselves
    # injected memory, and putting them in the memoryless baseline would
    # contaminate the contrast this project exists to measure. The cost is that our
    # numbers are not comparable to Li et al.'s -- and in the UNFAVOURABLE
    # direction, which is easy to misread. Their prompt supplies a clock-face
    # decoder (44% of our test items use clock phrasing), a cardinal decoder (22%;
    # 55% use one or the other) and ten worked examples. Ours supplies none of the
    # three.
    #
    # So: same frozen split, same grader, same decoding params, two extra prompts.
    #
    #   upstream-header  their decoders, still zero-shot. STILL A LEGITIMATE
    #                    memoryless baseline -- it is task specification, not a
    #                    worked solution. This is the honest comparability fix, and
    #                    the one to quote if only one is quoted.
    #   upstream         their prompt entire, header + 10 shots. Matches their
    #                    published protocol. NOT memoryless: never wire this into
    #                    refl-root, memp-build or any control arm.
    #
    # Read beside `ours` this isolates what the decoders are worth and what the
    # exemplars are worth on top of them. It does NOT change which protocol the
    # campaign runs -- `ours` stays the default everywhere else in this file.
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
    # Trial 1, shared by both arms. OUT is fixed per model: the auto-name carries
    # only split+timestamp, so three models' roots would differ only by when they ran.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      sub "sg-root-$tag" "$gpus" "$(t_of root "$tag")" \
        "$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=reflexion,PHASE=root,SPLIT=test,LIMIT=0,TRIALS=10,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=results/json/reflexion_root_stepgame_${tag}_test${PTAG}.json"
    done
    ;;

  refl-arms)
    # The control arm is not optional and matters MORE here than on SPaRC: nine
    # labels is a ~11% chance floor, so resampling at temp 0.6 converts failures on
    # its own and the cumulative curve slopes up with no memory at all
    # (docs/stepgame.md 4.1). Reflexion's effect is memory minus control at the
    # same trial index -- a retrieved-only curve is uninterpretable.
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
    # The memoryless TRAIN pass. Method-agnostic: MemoryBank distils from the same
    # file, which is what makes the two banks cover identical source items.
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
    # One job, one GPU, minutes: serves the EMBEDDING model, so it cannot share a
    # job with a chat phase. Depends only on the splits, so it can run in parallel
    # with build/distill. KEY_VARIANT is frozen into the .npz filename because a
    # mismatch between this and the evals is the one error the driver cannot
    # recover from silently.
    sub "sg-emb$KTAG" "A100:1" "01:00:00" \
      "$ENVX,$BENCHX,$KEYX,$PROTOX,DRIVER=memp,PHASE=embed,TP=1,OUT=$EMB"
    ;;

  memp-upgrade)
    # NOT a job: run_upgrade() takes no model and no GPU (on a single-turn benchmark
    # the verbatim record is the question plus the solver's worked derivation, built
    # by the benchmark), and serve_and_run.sbatch has no `upgrade` branch -- so
    # submitting it allocates GPUs, waits out the model load, and dies on
    # `unknown PHASE`. Writes <bank>_traj.json alongside each bank.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag _m _tp _g _e <<< "$spec"
      bank="results/json/memp_bank_stepgame_${tag}${PTAG}.json"
      [[ -f "$bank" ]] || { echo "skipping $tag: no $bank" >&2; continue; }
      echo "=== upgrade $tag ==="
      python scripts/run_memp.py --phase upgrade --benchmark stepgame \
        --stepgame-train "$TRAIN" --stepgame-test "$TEST" --bank "$bank"
    done
    ;;

  memp-eval)
    # Three arms per model. `fixed` is not a formality here -- it is THE hypothesis
    # (docs/stepgame.md 4.2): the optimal StepGame procedure is item-independent
    # ("map each sentence to a relation, chain source to target, sum the offsets,
    # read the sign"), so fixed ~= retrieved would mean procedural memory on this
    # task collapses to a single learned algorithm. `random` is the relevance
    # control that docs/memp.md 3.4 showed to be indispensable.
    #
    # Evals read the _traj bank (memp-upgrade's output) at GRANULARITY=trajectory:
    # docs/memp.md conclusion 5 found granularity the ONE intervention that produced
    # a large robust effect on SPaRC, and the compressed script was inert. Running
    # StepGame at script granularity only would replicate the inert configuration.
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
    # RULES_PREAMBLE=on only. On SPaRC the ablation was run both ways and was the
    # campaign's most informative NEGATIVE: the preamble made the case notes
    # dramatically better grounded and changed accuracy by nothing significant in
    # any of six paired tests (docs/memorybank.md conclusion 3). Re-running the off
    # variant here would spend six eval jobs re-answering a settled question; the
    # relation/offset table stands in for SPaRC's rulebook (4.4).
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      build="results/json/memp_build_stepgame_train_${tag}${PTAG}.json"
      # Same DEP_BUILD_<TAG>=<jobid> escape hatch memp-distill has, so MemoryBank
      # can be chained off an in-flight build rather than waiting for it to land.
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
        python scripts/run_reflexion.py --phase compare --memory "$mem" --control "$ctl" \
          --out "results/txt/reflexion_vs_control_stepgame_${tag}${PTAG}.txt"
      fi
      # Both cross-task methods are compared against the memoryless pass. The
      # baseline is the Reflexion root (trial 1) where it exists, as on SPaRC, so
      # every arm is paired against the identical 1000 items.
      base="$root"; [[ -f "$base" ]] || base="results/json/baseline_stepgame_${tag}_test${PTAG}.json"
      mret="results/json/memp_eval_stepgame_retrieved_test_${tag}${KTAG}.json"
      if [[ -f "$base" && -f "$mret" ]]; then
        echo "=== memp $tag ==="
        python scripts/run_memp.py --phase compare --baseline "$base" --retrieved "$mret" \
          $(f="results/json/memp_eval_stepgame_random_test_${tag}${KTAG}${PTAG}.json"; [[ -f "$f" ]] && echo --random "$f") \
          --out "results/txt/memp_vs_baseline_stepgame_${tag}${KTAG}${PTAG}.txt"
      fi
      bret="results/json/memorybank_eval_stepgame_retrieved_test_${tag}_rp.json"
      if [[ -f "$base" && -f "$bret" ]]; then
        echo "=== memorybank $tag ==="
        python scripts/run_memorybank.py --phase compare --baseline "$base" --retrieved "$bret" \
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
