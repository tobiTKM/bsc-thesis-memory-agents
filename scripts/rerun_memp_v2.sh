#!/bin/bash
# Re-run Memp on all three models with the rules-aware distillation prompt.
#
# WHY: docs/memp.md section 3.5 measured that the banks frequently misstate SPaRC's
# square/star rule, and section 3.6 concluded the LLM-written script therefore adds
# nothing on top of the mechanically-extracted trajectory. The cause was that neither
# distillation prompt ever contained the rulebook -- the distiller had to reverse-
# engineer the mechanics from cell labels and reliably guessed a Hamiltonian-path
# model instead of a region model. prompts.distill_rules_preamble now prepends the
# verbatim rules (sparc_rules, the same text the actor plays under, minus the
# `Final: <digit>` I/O contract). This script rebuilds the banks with that prompt and
# re-runs the two eval arms.
#
# WHAT IS REUSED, AND WHY IT IS SAFE TO REUSE:
#   * memp_build_train_*.json  -- the memoryless train pass. It uses
#     sparc_system_prompt, which this change leaves BYTE-IDENTICAL (verified), so a
#     re-run would burn ~11 wall-hours (91+141+482 min) to reproduce the same file.
#   * memp_embeddings_qwen3emb06b.npz -- depends only on the dataset and on
#     puzzle_key_text (key-fn puzzle-key-v1, unchanged). Shared by all three models.
# Set REBUILD=1 to re-run the train pass as well (it is not needed for this change).
#
# STAGE 1 (cheap, ~1 h total) rebuilds the three banks.
# STAGE 2 (expensive, ~9-19 h per model) re-runs the retrieved and random arms.
# Run stage 1, check the banks with scripts/audit_bank.py, and only then run stage 2:
# if the rule-error rate has not moved, the eval hours would measure nothing new.
#
#   bash scripts/rerun_memp_v2.sh distill     # stage 1
#   python scripts/audit_bank.py results/json/memp_bank_*_v2.json
#   bash scripts/rerun_memp_v2.sh eval        # stage 2
#   bash scripts/rerun_memp_v2.sh compare     # no GPU
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-}"
EMB=results/json/memp_embeddings_qwen3emb06b.npz

# The first attempt at this campaign (jobs 15369117/19/21, 2026-08-20) died in all
# three distill jobs with:
#   OSError: [Errno 122] Disk quota exceeded: '.../hf_cache/datasets/...lock'
# The shared ag_gipp PROJECT store was at 90.3% of its 7M-inode quota (our own subtree
# is 0.16% of that, so this is group-wide pressure we cannot fix). load_dataset() only
# needed to create a lock file and could not.
#
# The fix points the DATASET cache at the user's own vast-standard home (19 inodes,
# 9 MB) instead of the project store. Only HF_DATASETS_CACHE moves, deliberately:
# load_dataset runs host-side, outside apptainer, so its cache can live anywhere,
# whereas HF_HOME must stay under $AGPROJ because serve_and_run.sbatch's
# in_container() rewrites only paths under that prefix when it falls back to
# $HF_HOME/hub for a model not in the shared HUB (Qwen3-0.6B is exactly that case).
# Model weights therefore resolve exactly as before.
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
      # v2 in the name so the original banks stay on disk for the before/after audit.
      sbatch -J "memp-dist-${tag}-v2" -G "$gpus" -t 02:00:00 \
        --export=ALL,$ENVX,DRIVER=memp,PHASE=distill,MODEL="$model",TP="$tp",${extra:+$extra,}\
BUILD=results/json/memp_build_train_${tag}.json,\
OUT=results/json/memp_bank_${tag}_v2.json \
        scripts/serve_and_run.sbatch
    done
    echo "submitted 3 distill jobs; then: python scripts/audit_bank.py results/json/memp_bank_*_v2.json"
    ;;

  eval)
    # Round 1 of the v2 campaign, all ARM=retrieved / k=3 / granularity=script
    # (the sbatch defaults), six jobs:
    #   1-3  each model against its OWN v2 bank      -- the direct v1 replication
    #   4-5  both Qwen actors against gpt-oss's bank -- the section 3.7 transfer test,
    #        re-run now that the donor bank is not rule-wrong by construction
    #   6    gpt-oss against its own bank, gold entries filtered out (PROVENANCE=agent,
    #        71 entries) -- the paper-faithful successes-only build
    # No random arm this round (not requested). Worth noting for round 2: section 3.4
    # found retrieved == random at v1, but that was measured over content section 3.5
    # showed to be wrong ~half the time -- relevance cannot pay off when what is
    # retrieved is false. With corrected banks that contrast is newly informative.
    #
    # Chains on the distill jobs when the banks do not exist yet:
    #   DEP_QWEN06B=<jobid> DEP_QWEN32B=<jobid> DEP_GPTOSS=<jobid> bash ... eval
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
    # Round 2: proceduralization (script + verbatim trajectory) on the RULE-CORRECT
    # bank, gpt-oss only.
    #
    # Why this is the one run worth doing next. Round 1 showed the two components fix
    # DIFFERENT failure modes (gpt-oss, delta vs the 70/500 memoryless baseline):
    #     rule-correct script   solved  90   wrong-sol -23   dead-end  +2
    #     trajectory (v1)       solved 102   wrong-sol +37   dead-end -71
    #     proceduralization(v1) solved 103   wrong-sol +12   dead-end -46
    # The script buys rule compliance and nothing on navigation; the trajectory buys
    # navigation and costs rule compliance. v1's proceduralization stapled a
    # rule-WRONG script to the trajectory and landed at 103. If the components
    # compose, replacing that script with a rule-correct one should push
    # wrong-solutions below v1-proceduralization's 215 while holding dead-ends near
    # its 181, and beat 103. If it does not, they interfere -- equally worth knowing,
    # and it would close out the granularity question.
    #
    # Plain `trajectory` granularity is deliberately NOT re-run: format_procedures()
    # renders entry["trajectory"] and never entry["script"] at that setting, and the
    # trajectories are mechanically replayed from exemplars built off the unchanged
    # build file -- the v2 rendering is byte-identical to v1's 102/500. (The `verify`
    # stage below checks that claim rather than assuming it.)
    BANK=results/json/memp_bank_gptoss120b_v2_traj.json
    [[ -f "$BANK" ]] || { echo "missing $BANK -- run: run_memp.py --phase upgrade --bank memp_bank_gptoss120b_v2.json" >&2; exit 1; }
    sbatch -J memp-gran-proc-gptoss120b-v2 -G A100:4 -t 08:00:00 \
      --export=ALL,$ENVX,DRIVER=memp,PHASE=eval,MODEL=openai/gpt-oss-120b,TP=4,REASONING_PARSER=openai_gptoss,\
ARM=retrieved,LIMIT=0,GRANULARITY=proceduralization,BANK=$BANK,EMBEDDINGS=$EMB,\
OUT=results/json/memp_eval_gran_proceduralization_gptoss120b_v2.json \
      scripts/serve_and_run.sbatch
    echo "submitted proceduralization (gpt-oss, v2 rule-correct bank + trajectories)"
    ;;

  verify)
    # Is the v2 trajectory rendering really identical to v1's? If yes, the v1
    # trajectory arm (102/500) transfers as a reference point and needs no re-run.
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
    # Two contrasts, both paired over the same 500 puzzles:
    #   v1 -> v2  at the same model and arm. Identical build, embeddings, bank size,
    #             retriever, k, seed and block format: the distillation prompt is the
    #             only thing that moved, so this isolates the fix.
    #   v2 vs baseline, where the baseline is Reflexion's trial 1 -- the memoryless
    #             pass over the same split at the same seed. run_memp's
    #             baseline_outcomes() reads a reflexion root directly for exactly this
    #             reason; there is no separate baseline pass to run.
    python - <<'PY'
import pathlib, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_memp import eval_outcomes, baseline_outcomes, mcnemar_exact

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
    sed -n '2,30p' "$0"; exit 1 ;;
esac
