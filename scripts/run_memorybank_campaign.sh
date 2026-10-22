#!/bin/bash
# MemoryBank campaign on SPaRC: declarative memory (case note + verbatim record) for
# three models, two banks each (RULES_PREAMBLE=on|off). Needs Memp's
# memp_build_train_<tag>.json and embeddings; retrieval is identical to Memp's.
#
# Audit the banks before running eval:
#   bash scripts/run_memorybank_campaign.sh distill     # 6 jobs: builds the banks
#   python scripts/audit_bank.py results/json/memorybank_bank_*.json
#   bash scripts/run_memorybank_campaign.sh eval        # 12 jobs; needs the banks
#   bash scripts/run_memorybank_campaign.sh compare     # no GPU; after eval
set -euo pipefail
cd "$(dirname "$0")"/..

STAGE="${1:-}"

# Only the dataset cache leaves the inode-limited project store; HF_HOME must stay
# under $AGPROJ so serve_and_run.sbatch still finds the model weights.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/hf_datasets}"
[[ -d "$HF_DATASETS_CACHE" ]] || { echo "no dataset cache at $HF_DATASETS_CACHE" >&2; exit 1; }
ENVX="HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

EMB=results/json/memp_embeddings_qwen3emb06b.npz

# Eval walltimes are kept tight so jobs backfill; evals checkpoint, so an overrun
# resumes with RESUME=.
# tag           MODEL                    TP  GPUS       extra serve env         eval time
MODELS=(
  "qwen06b|Qwen/Qwen3-0.6B|1|A100:1||03:00:00"
  "qwen32b|Qwen/Qwen3-32B|2|A100:2||16:00:00"
  "gptoss120b|openai/gpt-oss-120b|4|A100:4|REASONING_PARSER=openai_gptoss|06:00:00"
)
# on -> rp, off -> norp; the variant tag is in every bank and eval filename.
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
    # Retrieved (top-3) and seeded-random arms per bank. k, seed and granularity are
    # the sbatch defaults, which match Memp's.
    #   DEP_<TAG>_<VTAG>=<jobid> chains on a distill job whose bank does not exist yet.
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
    # Paired contrasts: vs baseline (Reflexion trial 1), vs random, and vs Memp's
    # retrieved arm when present.
    for spec in "${MODELS[@]}"; do
      IFS='|' read -r tag _m _tp _g _e _t <<< "$spec"
      for v in "${VARIANTS[@]}"; do
        IFS='|' read -r _rp vtag <<< "$v"
        ret="results/json/memorybank_eval_retrieved_test_${tag}_${vtag}.json"
        rnd="results/json/memorybank_eval_random_test_${tag}_${vtag}.json"
        [[ -f "$ret" ]] || continue
        memp_arm="results/json/memp_eval_retrieved_test_${tag}_v2.json"
        echo "=== $tag / $vtag ==="
        python scripts/run.py memorybank --phase compare \
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
