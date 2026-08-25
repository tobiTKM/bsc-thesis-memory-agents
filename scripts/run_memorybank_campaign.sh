#!/bin/bash
# The MemoryBank campaign: declarative memory over SPaRC, three models, two banks each.
#
# Mirrors rerun_memp_v2.sh's structure because it mirrors its experiment: the same
# three actors, the same 500 test puzzles, the same seed and sampling, the same
# baselines (each model's Reflexion trial 1). What differs is the content of the
# injected memory -- a past CASE (concrete recollection + verbatim record) instead of
# a PROCEDURE -- which is precisely the comparison the thesis's sub-question 2 asks
# for. See docs/memorybank.md.
#
# WHAT IS REUSED, AND WHY IT IS SAFE TO REUSE:
#   * memp_build_train_<tag>.json -- the memoryless train pass. Method-agnostic
#     content (it is just "what happened when this model attempted the train split
#     with no memory"), already paid for at ~8 GPU-h per model. Both cross-task
#     methods distil from the same file, which also means their banks cover the
#     identical 500 source puzzles.
#   * memp_embeddings_qwen3emb06b.npz -- depends only on the dataset and on
#     puzzle_key_text (key-fn puzzle-key-v1, unchanged). MemoryBankMemory subclasses
#     MempMemory, so retrieval is byte-identical between the two methods by
#     construction; that is what makes the cross-method contrast single-factor.
#
# THE TWO BANKS PER MODEL (the rules-preamble ablation, RULES_PREAMBLE=on|off):
#   docs/memp.md section 3.5 found Memp's v1 banks misstated SPaRC's square/star rule
#   in 33-51% of entries because no distillation prompt contained the rulebook;
#   prepending it fixed that (gpt-oss: 50.7% -> 99.8% framed correctly). Whether a
#   *case note* -- which recounts an instance rather than asserting a rule -- needs
#   the same help is genuinely open, and the paper's own instruction is the plain
#   "summarize the events and key information". So both are built and measured.
#
# STAGE 1 (cheap, ~20 min/job) builds the six banks.
# STAGE 2 (expensive, ~2-11 h per job) runs the eval arms.
# Run stage 1, AUDIT THE BANKS, and only then run stage 2: if the case notes are
# rule-broken at v1-Memp rates, or if the two variants are indistinguishable, the
# eval hours would measure the wrong thing or a duplicate.
#
#   bash scripts/run_memorybank_campaign.sh distill     # stage 1: 6 jobs
#   python scripts/audit_bank.py results/json/memorybank_bank_*.json
#   bash scripts/run_memorybank_campaign.sh eval        # stage 2: 12 jobs
#   bash scripts/run_memorybank_campaign.sh compare     # no GPU
set -euo pipefail
cd "$(dirname "$0")"/..

STAGE="${1:-}"

# The ag_gipp PROJECT store sits near its 7M-inode quota, which makes load_dataset's
# lock file creation fail (see rerun_memp_v2.sh for the full diagnosis). Only the
# DATASET cache moves to the user's own home; HF_HOME must stay under $AGPROJ so
# serve_and_run.sbatch's in_container() path rewriting still finds model weights.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/vast-standard/home/t.mark/u21167/hf_datasets}"
[[ -d "$HF_DATASETS_CACHE" ]] || { echo "no dataset cache at $HF_DATASETS_CACHE" >&2; exit 1; }
ENVX="HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

EMB=results/json/memp_embeddings_qwen3emb06b.npz

# tag           MODEL                    TP  GPUS       extra serve env         eval time
MODELS=(
  "qwen06b|Qwen/Qwen3-0.6B|1|A100:1||04:00:00"
  "qwen32b|Qwen/Qwen3-32B|2|A100:2||12:00:00"
  "gptoss120b|openai/gpt-oss-120b|4|A100:4|REASONING_PARSER=openai_gptoss|06:00:00"
)
# on -> rp, off -> norp. The variant is in every bank and eval filename: two banks
# per model exist at once and a job names its bank on the submit line, where a wrong
# pick would stay invisible until the numbers disagreed.
VARIANTS=("on|rp" "off|norp")

case "$STAGE" in
  distill)
    for spec in "${MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra _t <<< "$spec"
      for v in "${VARIANTS[@]}"; do
        IFS='|' read -r rp vtag <<< "$v"
        sbatch -J "mb-dist-${tag}-${vtag}" -G "$gpus" -t 02:00:00 \
          --export=ALL,$ENVX,DRIVER=memorybank,PHASE=distill,MODEL="$model",TP="$tp",${extra:+$extra,}\
RULES_PREAMBLE=$rp,\
BUILD=results/json/memp_build_train_${tag}.json,\
OUT=results/json/memorybank_bank_${tag}_${vtag}.json \
          scripts/serve_and_run.sbatch
      done
    done
    echo "submitted 6 distill jobs (3 models x rules-preamble on/off)"
    echo "next: python scripts/audit_bank.py results/json/memorybank_bank_*.json"
    ;;

  eval)
    # Two arms per bank: cosine top-k=3, and the seeded-random draw over the same
    # bank. Random is not optional here -- docs/memp.md section 3.4 found retrieved
    # and random statistically indistinguishable at every configuration tested, so a
    # retrieved-only result would be uninterpretable. Everything else is the sbatch
    # default and matches Memp: k=3, seed 0, granularity case-record.
    #
    # Chains on the distill jobs when a bank does not exist yet:
    #   DEP_<TAG>_<VTAG>=<jobid> bash scripts/run_memorybank_campaign.sh eval
    submit() {   # name gpus tlimit model tp extra arm bank out [depjob]
      local name=$1 gpus=$2 tlimit=$3 model=$4 tp=$5 extra=$6 arm=$7 bank=$8 out=$9 dep=${10:-}
      sbatch -J "$name" -G "$gpus" -t "$tlimit" ${dep:+--dependency=afterok:$dep} \
        --export=ALL,$ENVX,DRIVER=memorybank,PHASE=eval,MODEL="$model",TP="$tp",${extra:+$extra,}\
ARM="$arm",LIMIT=0,BANK="$bank",EMBEDDINGS=$EMB,OUT="$out" \
        scripts/serve_and_run.sbatch
    }
    for spec in "${MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra tlimit <<< "$spec"
      for v in "${VARIANTS[@]}"; do
        IFS='|' read -r _rp vtag <<< "$v"
        bank="results/json/memorybank_bank_${tag}_${vtag}.json"
        depvar="DEP_$(echo "${tag}_${vtag}" | tr '[:lower:]' '[:upper:]')"
        dep=""
        [[ -f "$bank" ]] || dep="${!depvar:-}"
        [[ -f "$bank" || -n "$dep" ]] || {
          echo "skipping ${tag}/${vtag}: no $bank and no $depvar=<jobid>" >&2; continue; }
        for arm in retrieved random; do
          submit "mb-${arm:0:3}-${tag}-${vtag}" "$gpus" "$tlimit" "$model" "$tp" \
                 "$extra" "$arm" "$bank" \
                 "results/json/memorybank_eval_${arm}_test_${tag}_${vtag}.json" "$dep"
        done
      done
    done
    echo "submitted the eval grid (models x preamble variants x {retrieved,random})"
    ;;

  compare)
    # Three contrasts per model, all paired over the same 500 puzzles:
    #   vs baseline  -- the memoryless pass (Reflexion trial 1, read directly)
    #   vs random    -- does RELEVANCE do the work, or just having a case in context
    #   vs Memp      -- declarative vs procedural, the sub-question-2 contrast, run
    #                   against Memp's best arm for that model where one exists
    for spec in "${MODELS[@]}"; do
      IFS='|' read -r tag _m _tp _g _e _t <<< "$spec"
      for v in "${VARIANTS[@]}"; do
        IFS='|' read -r _rp vtag <<< "$v"
        ret="results/json/memorybank_eval_retrieved_test_${tag}_${vtag}.json"
        rnd="results/json/memorybank_eval_random_test_${tag}_${vtag}.json"
        [[ -f "$ret" ]] || continue
        memp_arm="results/json/memp_eval_retrieved_test_${tag}_v2.json"
        echo "=== $tag / $vtag ==="
        python scripts/run_memorybank.py --phase compare \
          --baseline "results/json/reflexion_root_${tag}_test.json" \
          --retrieved "$ret" ${rnd:+$([[ -f "$rnd" ]] && echo --random "$rnd")} \
          $([[ -f "$memp_arm" ]] && echo --memp "$memp_arm") \
          --out "results/txt/memorybank_vs_baseline_${tag}_${vtag}.txt"
      done
    done
    ;;

  *)
    echo "usage: bash scripts/run_memorybank_campaign.sh {distill|eval|compare}" >&2
    exit 1 ;;
esac
