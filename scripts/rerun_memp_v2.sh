#!/bin/bash
# Re-run Memp on SPaRC for three models with the rules-aware distillation prompt
# (prompts.distill_rules_preamble). Reuses the existing train pass and embeddings.
#
#   bash scripts/rerun_memp_v2.sh distill    # 3 jobs: banks as memp_bank_*_v2.json
#   python scripts/audit_bank.py results/json/memp_bank_*_v2.json   # check before eval
#   bash scripts/rerun_memp_v2.sh eval       # 6 jobs; DEP_<MODEL>=<jobid> chains on distill
#   bash scripts/rerun_memp_v2.sh proc       # 1 job; needs memp_bank_gptoss120b_v2_traj.json
#   bash scripts/rerun_memp_v2.sh randtraj   # 1 job: random-trajectory control
#   bash scripts/rerun_memp_v2.sh verify     # no GPU
#   bash scripts/rerun_memp_v2.sh compare    # no GPU
# DEP_<MODEL> is DEP_QWEN06B, DEP_QWEN32B or DEP_GPTOSS.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-}"
EMB=results/json/memp_embeddings_qwen3emb06b.npz

# The ag_gipp project store is near its inode quota, which breaks load_dataset's lock
# file. Only the dataset cache moves: HF_HOME must stay under $AGPROJ, the only prefix
# serve_and_run.sbatch rewrites for the container.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/vast-standard/home/t.mark/u21167/hf_datasets}"
[[ -d "$HF_DATASETS_CACHE" ]] || { echo "no dataset cache at $HF_DATASETS_CACHE" >&2; exit 1; }
ENVX="HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

# tag           MODEL                    TP  GPUS       extra serve env         eval time
MODELS=(
  "qwen06b|Qwen/Qwen3-0.6B|1|A100:1||04:00:00"
  "qwen32b|Qwen/Qwen3-32B|2|A100:2||12:00:00"
  "gptoss120b|openai/gpt-oss-120b|4|A100:4|REASONING_PARSER=openai_gptoss|06:00:00"
)

case "$STAGE" in
  distill)
    for spec in "${MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra _t <<< "$spec"
      # v2 suffix keeps the v1 banks on disk for the before/after audit.
      sbatch -J "memp-dist-${tag}-v2" -G "$gpus" -t 02:00:00 \
        --export=ALL,$ENVX,DRIVER=memp,PHASE=distill,MODEL="$model",TP="$tp",${extra:+$extra,}\
BUILD=results/json/memp_build_train_${tag}.json,\
OUT=results/json/memp_bank_${tag}_v2.json \
        scripts/serve_and_run.sbatch
    done
    echo "submitted 3 distill jobs; then: python scripts/audit_bank.py results/json/memp_bank_*_v2.json"
    ;;

  eval)
    # All ARM=retrieved, k=3, script granularity (the sbatch defaults).
    submit() {   # name gpus tlimit model tp extra bank out [depjob]
      local name=$1 gpus=$2 tlimit=$3 model=$4 tp=$5 extra=$6 bank=$7 out=$8 dep=${9:-}
      sbatch -J "$name" -G "$gpus" -t "$tlimit" ${dep:+--dependency=afterok:$dep} \
        --export=ALL,$ENVX,DRIVER=memp,PHASE=eval,MODEL="$model",TP="$tp",${extra:+$extra,}\
ARM=retrieved,LIMIT=0,BANK="$bank",EMBEDDINGS=$EMB,OUT="$out" \
        scripts/serve_and_run.sbatch
    }
    need_dep() {  # echo the dep jobid only if the bank is still missing
      [[ -f "$1" ]] && echo "" || echo "$2"
    }
    B06=results/json/memp_bank_qwen06b_v2.json
    B32=results/json/memp_bank_qwen32b_v2.json
    BOSS=results/json/memp_bank_gptoss120b_v2.json
    R=results/json/memp_eval_retrieved_test

    # 1-3: own bank
    submit memp-ret-qwen06b-v2    A100:1 04:00:00 Qwen/Qwen3-0.6B    1 "" \
           "$B06"  "${R}_qwen06b_v2.json"    "$(need_dep "$B06" "${DEP_QWEN06B:-}")"
    submit memp-ret-qwen32b-v2    A100:2 12:00:00 Qwen/Qwen3-32B     2 "" \
           "$B32"  "${R}_qwen32b_v2.json"    "$(need_dep "$B32" "${DEP_QWEN32B:-}")"
    submit memp-ret-gptoss120b-v2 A100:4 06:00:00 openai/gpt-oss-120b 4 REASONING_PARSER=openai_gptoss \
           "$BOSS" "${R}_gptoss120b_v2.json" "$(need_dep "$BOSS" "${DEP_GPTOSS:-}")"
    # 4-5: transfer -- Qwen actors reading gpt-oss's bank
    submit memp-xfer-oss2qwen06b-v2 A100:1 04:00:00 Qwen/Qwen3-0.6B 1 "" \
           "$BOSS" "${R}_qwen06b_bank-gptoss_v2.json" "$(need_dep "$BOSS" "${DEP_GPTOSS:-}")"
    submit memp-xfer-oss2qwen32b-v2 A100:2 12:00:00 Qwen/Qwen3-32B  2 "" \
           "$BOSS" "${R}_qwen32b_bank-gptoss_v2.json" "$(need_dep "$BOSS" "${DEP_GPTOSS:-}")"
    # 6: agent-provenance only (no gold-path hindsight entries)
    sbatch -J memp-agentonly-gptoss120b-v2 -G A100:4 -t 06:00:00 \
      ${DEP_GPTOSS:+$([[ -f "$BOSS" ]] || echo --dependency=afterok:$DEP_GPTOSS)} \
      --export=ALL,$ENVX,DRIVER=memp,PHASE=eval,MODEL=openai/gpt-oss-120b,TP=4,REASONING_PARSER=openai_gptoss,\
ARM=retrieved,LIMIT=0,PROVENANCE=agent,BANK=$BOSS,EMBEDDINGS=$EMB,\
OUT=${R}_gptoss120b_agentonly_v2.json \
      scripts/serve_and_run.sbatch
    echo "submitted 6 eval jobs (k=3, script, retrieved)"
    ;;

  proc)
    # Proceduralization (script + trajectory) on the v2 bank, gpt-oss only. Plain
    # `trajectory` is not re-run: it never renders the script, so it equals v1.
    BANK=results/json/memp_bank_gptoss120b_v2_traj.json
    [[ -f "$BANK" ]] || { echo "missing $BANK -- run: scripts/run.py memp --phase upgrade --bank memp_bank_gptoss120b_v2.json" >&2; exit 1; }
    sbatch -J memp-gran-proc-gptoss120b-v2 -G A100:4 -t 08:00:00 \
      --export=ALL,$ENVX,DRIVER=memp,PHASE=eval,MODEL=openai/gpt-oss-120b,TP=4,REASONING_PARSER=openai_gptoss,\
ARM=retrieved,LIMIT=0,GRANULARITY=proceduralization,BANK=$BANK,EMBEDDINGS=$EMB,\
OUT=results/json/memp_eval_gran_proceduralization_gptoss120b_v2.json \
      scripts/serve_and_run.sbatch
    echo "submitted proceduralization (gpt-oss, v2 rule-correct bank + trajectories)"
    ;;

  randtraj)
    # Random-trajectory control on the v1 _traj bank, the same file as the retrieved
    # trajectory run, so only the arm differs (`verify` shows v1 and v2 render alike).
    BANK=results/json/memp_bank_gptoss120b_traj.json
    [[ -f "$BANK" ]] || { echo "missing $BANK" >&2; exit 1; }
    sbatch -J memp-randtraj-gptoss120b -G A100:4 -t 08:00:00 \
      --export=ALL,$ENVX,DRIVER=memp,PHASE=eval,MODEL=openai/gpt-oss-120b,TP=4,REASONING_PARSER=openai_gptoss,\
ARM=random,LIMIT=0,GRANULARITY=trajectory,BANK=$BANK,EMBEDDINGS=$EMB,\
OUT=results/json/memp_eval_gran_trajectory_random_gptoss120b.json \
      scripts/serve_and_run.sbatch
    echo "submitted random-trajectory control (gpt-oss, same bank as the retrieved run)"
    ;;

  verify)
    # Checks that the v1 and v2 _traj banks share trajectories and differ in scripts.
    /mnt/ceph-hdd/projects/ag_gipp/u21167/.venv_client/bin/python - <<'PY'
import json, pathlib
J = pathlib.Path("results/json")
a = {e["source_pid"]: e for e in json.loads((J/"memp_bank_gptoss120b_traj.json").read_text())["entries"]}
b = {e["source_pid"]: e for e in json.loads((J/"memp_bank_gptoss120b_v2_traj.json").read_text())["entries"]}
common = set(a) & set(b)
same_traj = sum(1 for p in common if a[p].get("trajectory") == b[p].get("trajectory"))
same_script = sum(1 for p in common if a[p].get("script") == b[p].get("script"))
print(f"  entries compared:        {len(common)}")
print(f"  identical trajectory:    {same_traj}/{len(common)}  <- expect all: mechanically replayed")
print(f"  identical script:        {same_script}/{len(common)}  <- expect ~0: this is the change")
PY
    ;;

  compare)
    # Both paired over the same puzzles: v1 -> v2 at the same model and arm (only the
    # distillation prompt differs), and v2 vs the Reflexion root (trial 1) as baseline.
    python - <<'PY'
import pathlib, sys
sys.path.insert(0, "src")
from memory_agents.methods.memp import eval_outcomes, baseline_outcomes, mcnemar_exact

def paired(a_wins, b_wins, a_label, b_label):
    common = sorted(set(a_wins) & set(b_wins))
    b_only = sum(1 for p in common if b_wins[p] and not a_wins[p])
    a_only = sum(1 for p in common if a_wins[p] and not b_wins[p])
    return (len(common), sum(a_wins[p] for p in common), sum(b_wins[p] for p in common),
            a_only, b_only, mcnemar_exact(b_only, a_only))

# label -> (v1 file or None, v2 file)
VARIANTS = {
    "qwen06b":    [("own bank",  "memp_eval_retrieved_test_qwen06b.json",
                                 "memp_eval_retrieved_test_qwen06b_v2.json"),
                   ("gpt-oss bank", "memp_eval_transfer_oss2qwen06b.json",
                                 "memp_eval_retrieved_test_qwen06b_bank-gptoss_v2.json")],
    "qwen32b":    [("own bank",  "memp_eval_retrieved_test_qwen32b.json",
                                 "memp_eval_retrieved_test_qwen32b_v2.json"),
                   ("gpt-oss bank", None,
                                 "memp_eval_retrieved_test_qwen32b_bank-gptoss_v2.json")],
    "gptoss120b": [("own bank",  "memp_eval_retrieved_test_gptoss120b.json",
                                 "memp_eval_retrieved_test_gptoss120b_v2.json"),
                   ("agent-only", "memp_eval_agentonly_gptoss120b.json",
                                 "memp_eval_retrieved_test_gptoss120b_agentonly_v2.json")],
}
J = "results/json/"
for tag, rows in VARIANTS.items():
    root = pathlib.Path(f"{J}reflexion_root_{tag}_test.json")
    base_wins, _ = baseline_outcomes(root) if root.exists() else ({}, {})
    print(f"\n=============== {tag}  (baseline = reflexion trial 1: "
          f"{sum(base_wins.values())}/{len(base_wins)}) ===============")
    for label, v1, v2 in rows:
        p2 = pathlib.Path(J + v2)
        if not p2.exists():
            print(f"  {label:<14} v2 not finished yet"); continue
        n_wins, _ = eval_outcomes(p2)
        if base_wins:
            n, sb, sn, bo, no, p = paired(base_wins, n_wins, "base", "v2")
            print(f"  {label:<14} baseline {sb:>3}/{n} -> v2 {sn:>3}/{n}   "
                  f"(v2 alone {no}, base alone {bo}, p={p:.4f})")
        if v1 and pathlib.Path(J + v1).exists():
            o_wins, _ = eval_outcomes(J + v1)
            n, so, sn, oo, no, p = paired(o_wins, n_wins, "v1", "v2")
            print(f"  {'':<14} v1       {so:>3}/{n} -> v2 {sn:>3}/{n}   "
                  f"(v2 alone {no}, v1 alone {oo}, p={p:.4f})")
PY
    ;;

  *)
    awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 1 ;;
esac
