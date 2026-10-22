#!/bin/bash
# Contrastive memory campaign: DOs distilled from the actor's successes, DON'Ts from
# its failures. BENCH=sparc (default) or BENCH=stepgame selects the benchmark.
#
#   bash scripts/run_contrastive_campaign.sh distill       # 1 job per model: builds the bank
#   bash scripts/run_contrastive_campaign.sh eval          # 5 arms per model; needs the bank
#   bash scripts/run_contrastive_campaign.sh exemplars     # no GPU (login node): bank -> _ex bank with worked examples
#   bash scripts/run_contrastive_campaign.sh wave2         # 4 exemplar/framing arms; needs the _ex bank
#   bash scripts/run_contrastive_campaign.sh standard      # 3 arms at the standard config (exemplars on)
#   bash scripts/run_contrastive_campaign.sh balanced-fix  # re-runs the balanced arm into a _v2 file
#   bash scripts/run_contrastive_campaign.sh compare       # no GPU; after eval
#   bash scripts/run_contrastive_campaign.sh status        # lists banks and eval results
#
# MODELS="qwen32b" overrides the default actor (sparc: gptoss120b, stepgame: qwen32b).
# EXCLUDE="ggpu195" keeps jobs off named nodes; ONLY="hbo aex" restricts wave2/standard
# to those arm infixes. DEP_DISTILL_<TAG>=<jobid> chains eval/standard off a distill job.
# DRYRUN=1 prints the sbatch lines instead of submitting; check the file paths after any edit.
set -euo pipefail
cd "$(dirname "$0")"/..

STAGE="${1:-}"
BENCH="${BENCH:-sparc}"

# The ag_gipp project store is near its inode quota, which makes load_dataset's lock
# file creation fail; keep the dataset cache elsewhere.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/hf_datasets}"
[[ -d "$HF_DATASETS_CACHE" ]] || { echo "no dataset cache at $HF_DATASETS_CACHE" >&2; exit 1; }
ENVX="HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

case "$BENCH" in
  sparc)
    BENCHX="BENCHMARK=sparc"
    PTAG=""
    BM=""
    EMB="results/json/memp_embeddings_qwen3emb06b.npz"
    BUILD_FMT="results/json/memp_build_train_%s.json"
    DEFAULT_MODELS="gptoss120b"
    ;;
  stepgame)
    SPLITS=data/stepgame/splits
# Must match run_stepgame_campaign.sh, whose build file the bank is distilled from;
# a different test split would leave the arms unpaired.
    TEST="${TEST:-$SPLITS/stepgame_clean_test_k1-10_n1000.jsonl}"
    TRAIN="${TRAIN:-$SPLITS/stepgame_clean_train_byhops_k1-5_n500.jsonl}"
    PROMPT_PROTOCOL="${PROMPT_PROTOCOL:-upstream-header}"
    BENCHX="BENCHMARK=stepgame,STEPGAME_TRAIN=$TRAIN,STEPGAME_TEST=$TEST,PROMPT_PROTOCOL=$PROMPT_PROTOCOL"
    PTAG=""; [[ "$PROMPT_PROTOCOL" == "ours" ]] || PTAG="_$PROMPT_PROTOCOL"
    BM="stepgame_"
    BENCH_FLAGS="--stepgame-train $TRAIN --stepgame-test $TEST --prompt-protocol $PROMPT_PROTOCOL"
    EMB="results/json/memp_embeddings_stepgame.npz"
    BUILD_FMT="results/json/memp_build_stepgame_train_%s${PTAG}.json"
    DEFAULT_MODELS="qwen32b"
    ;;
  *) echo "BENCH must be sparc or stepgame (got '$BENCH')" >&2; exit 1 ;;
esac
KEYX="KEY_VARIANT=${KEY_VARIANT:-structural}"

# tag|model|TP|gpus|extra serve env, as in rerun_memp_v2.sh.
ALL_MODELS=(
  "qwen06b|Qwen/Qwen3-0.6B|1|A100:1|"
  "qwen32b|Qwen/Qwen3-32B|2|A100:2|"
  "gptoss120b|openai/gpt-oss-120b|4|A100:4|REASONING_PARSER=openai_gptoss"
)
keep=()
for spec in "${ALL_MODELS[@]}"; do
  for want in ${MODELS:-$DEFAULT_MODELS}; do
    [[ "${spec%%|*}" == "$want" ]] && keep+=("$spec")
  done
done
[[ ${#keep[@]} -gt 0 ]] || { echo "MODELS matched no known tag" >&2; exit 1; }
ALL_MODELS=("${keep[@]}")

# Walltimes are sized from observed runtimes: an oversized limit blocks backfill, and
# every phase checkpoints, so an overrun resumes with RESUME=<file>.
t_of() {   # t_of <stage> <tag>
  case "$BENCH:$1:$2" in
    sparc:distill:qwen06b)       echo 03:00:00 ;;
    sparc:distill:qwen32b)       echo 03:00:00 ;;
    sparc:distill:gptoss120b)    echo 05:00:00 ;;
    sparc:eval:qwen06b)          echo 03:00:00 ;;
    sparc:eval:qwen32b)          echo 14:00:00 ;;
    sparc:eval:gptoss120b)       echo 06:00:00 ;;
    stepgame:distill:qwen06b)    echo 02:00:00 ;;
    stepgame:distill:qwen32b)    echo 04:00:00 ;;
    stepgame:distill:gptoss120b) echo 03:00:00 ;;
    stepgame:eval:qwen06b)       echo 03:00:00 ;;
    stepgame:eval:qwen32b)       echo 08:00:00 ;;
    stepgame:eval:gptoss120b)    echo 05:00:00 ;;
    *) echo "no walltime for $BENCH stage '$1' tag '$2'" >&2; exit 1 ;;
  esac
}

bank_of() { printf 'results/json/contrastive_bank_%s%s%s%s.json' "$BM" "$1" "$PTAG" "${2:-}"; }

# A draining node with a failed health check can still accept jobs and kill them within
# seconds, with no log. `sinfo -p scc-gpu -t drain` lists such nodes.
EXCL=""
[[ -n "${EXCLUDE:-}" ]] && EXCL="--exclude=$(echo "$EXCLUDE" | tr ' ' ',')"

# Benchmark prefix: both benchmarks share arm and actor tags but write different files.
JPFX="co"; [[ "$BENCH" == "stepgame" ]] && JPFX="cosg"

sub() {   # sub <jobname-suffix> <gpus> <time> <dep-or-empty> <exports...>
  local name="$JPFX-$1" gpus=$2 tlimit=$3 dep=$4; shift 4
  if [[ "${DRYRUN:-0}" == "1" ]]; then
    printf 'sbatch -J %s -G %s -t %s %s%s--export=ALL,%s scripts/serve_and_run.sbatch\n' \
           "$name" "$gpus" "$tlimit" "${dep:+--dependency=afterok:$dep }" \
           "${EXCL:+$EXCL }" "$*"
  else
    sbatch -J "$name" -G "$gpus" -t "$tlimit" ${dep:+--dependency=afterok:$dep} \
      $EXCL --export=ALL,"$*" scripts/serve_and_run.sbatch
  fi
}

# granularity|arm|jobname-infix. `hindsight` is the matched control for `avoid`; `balanced`
# guarantees DO entries in the block, which a plain top-3 rarely contains.
ARMS=(
  "avoid|retrieved|avo"
  "hindsight|retrieved|hin"
  "positive-only|retrieved|pos"
  "avoid|balanced|bal"
  "avoid|random|rnd"
)

# Exemplar x framing 2x2. Uses the _ex bank, whose entry texts are identical to the
# eval-stage bank; only the attached worked example differs.
# gran|arm|exemplars|framing|infix
WAVE2=(
  "hindsight|retrieved|on|mixed|hex"     # exemplar only
  "hindsight|retrieved|off|matched|hmf"  # framing only
  "hindsight|retrieved|on|matched|hbo"   # both
  "avoid|retrieved|on|mixed|aex"         # avoid with exemplar
)

case "$STAGE" in
  distill)
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      build=$(printf "$BUILD_FMT" "$tag")
      [[ -f "$build" ]] || { echo "skipping $tag: no $build" >&2; continue; }
      sub "dist-$tag" "$gpus" "$(t_of distill "$tag")" "" \
        "$ENVX,$BENCHX,$KEYX,DRIVER=contrastive,PHASE=distill,BUILD=$build,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=$(bank_of "$tag")"
    done
    ;;

  eval)
    [[ -f "$EMB" ]] || { echo "no $EMB (run the memp embed phase first)" >&2; exit 1; }
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      bank=$(bank_of "$tag")
      depvar="DEP_DISTILL_$(echo "$tag" | tr '[:lower:]-' '[:upper:]_')"
      dep="${!depvar:-}"
      [[ -f "$bank" || -n "$dep" ]] || {
        echo "skipping $tag: no $bank and no $depvar=<jobid>" >&2; continue; }
      for a in "${ARMS[@]}"; do
        IFS='|' read -r gran arm infix <<< "$a"
        out="results/json/contrastive_eval_${BM}${gran}_${arm}_test_${tag}${PTAG}.json"
        sub "$infix-$tag" "$gpus" "$(t_of eval "$tag")" "$dep" \
          "$ENVX,$BENCHX,$KEYX,DRIVER=contrastive,PHASE=eval,CO_ARM=$arm,CO_GRANULARITY=$gran,MIN_POSITIVE=${MIN_POSITIVE:-1},LIMIT=0,BANK=$bank,EMBEDDINGS=$EMB,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=$out"
      done
    done
    ;;

  exemplars)
    # CPU only; runs on the login node.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      bank=$(bank_of "$tag"); build=$(printf "$BUILD_FMT" "$tag")
      [[ -f "$bank" ]] || { echo "skipping $tag: no $bank" >&2; continue; }
      python scripts/run.py contrastive --phase exemplars --benchmark "$BENCH" \
        ${BENCH_FLAGS:-} --bank "$bank" --build "$build" \
        --out "$(bank_of "$tag" _ex)"
    done
    ;;

  wave2)
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      bank=$(bank_of "$tag" _ex)
      [[ -f "$bank" ]] || { echo "skipping $tag: no $bank (run the exemplars stage)" >&2; continue; }
      for a in "${WAVE2[@]}"; do
        IFS='|' read -r gran arm ex fr infix <<< "$a"
        if [[ -n "${ONLY:-}" ]] && [[ " ${ONLY} " != *" $infix "* ]]; then continue; fi
        sfx=""; [[ "$ex" == "on" ]] && sfx="${sfx}_ex"
        [[ "$fr" == "matched" ]] && sfx="${sfx}_mf"
        out="results/json/contrastive_eval_${BM}${gran}_${arm}_test_${tag}${PTAG}${sfx}.json"
        sub "$infix-$tag" "$gpus" "$(t_of eval "$tag")" "" \
          "$ENVX,$BENCHX,$KEYX,DRIVER=contrastive,PHASE=eval,CO_ARM=$arm,CO_GRANULARITY=$gran,CO_EXEMPLARS=$ex,CO_FRAMING=$fr,MIN_POSITIVE=${MIN_POSITIVE:-1},LIMIT=0,BANK=$bank,EMBEDDINGS=$EMB,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=$out"
      done
    done
    ;;

  standard)
    # Standard configuration: the worked exemplar attached to every arm.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      # Newer banks carry the exemplars; older ones have them in a sibling _ex file.
      bank=$(bank_of "$tag" _ex); [[ -f "$bank" ]] || bank=$(bank_of "$tag")
      depvar="DEP_DISTILL_$(echo "$tag" | tr '[:lower:]-' '[:upper:]_')"; dep="${!depvar:-}"
      [[ -f "$bank" || -n "$dep" ]] || { echo "skipping $tag: no bank and no $depvar=<jobid>" >&2; continue; }
      for a in "avoid|retrieved|ret" "avoid|random|rnd" \
               "positive-only|retrieved|pos"; do
        IFS='|' read -r gran arm infix <<< "$a"
        if [[ -n "${ONLY:-}" ]] && [[ " ${ONLY} " != *" $infix "* ]]; then continue; fi
        fr=mixed; sfx=_ex
        # framing=matched only applies to `hindsight`, whose items are all DO.
        [[ "$gran" == "hindsight" ]] && { fr=matched; sfx=_ex_mf; }
        out="results/json/contrastive_eval_${BM}${gran}_${arm}_test_${tag}${PTAG}${sfx}.json"
        [[ -f "$out" && "${FORCE:-0}" != "1" ]] && { echo "skipping $infix-$tag: $out exists (FORCE=1 to redo)" >&2; continue; }
        sub "s${infix}-$tag" "$gpus" "$(t_of eval "$tag")" "$dep" \
          "$ENVX,$BENCHX,$KEYX,DRIVER=contrastive,PHASE=eval,CO_ARM=$arm,CO_GRANULARITY=$gran,CO_EXEMPLARS=on,CO_FRAMING=$fr,MIN_POSITIVE=${MIN_POSITIVE:-1},LIMIT=0,BANK=$bank,EMBEDDINGS=$EMB,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=$out"
      done
    done
    ;;

  balanced-fix)
    # Re-runs `balanced` with similarity-ordered blocks; the _v2 suffix keeps the earlier output.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      bank=$(bank_of "$tag")
      [[ -f "$bank" ]] || { echo "skipping $tag: no $bank" >&2; continue; }
      out="results/json/contrastive_eval_${BM}avoid_balanced_test_${tag}${PTAG}_v2.json"
      sub "bal2-$tag" "$gpus" "$(t_of eval "$tag")" "" \
        "$ENVX,$BENCHX,$KEYX,DRIVER=contrastive,PHASE=eval,CO_ARM=balanced,CO_GRANULARITY=avoid,MIN_POSITIVE=${MIN_POSITIVE:-1},LIMIT=0,BANK=$bank,EMBEDDINGS=$EMB,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=$out"
    done
    ;;

  compare)
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag _m _tp _g _e <<< "$spec"
      e() { printf 'results/json/contrastive_eval_%s%s_%s_test_%s%s.json' \
                   "$BM" "$1" "$2" "$tag" "$PTAG"; }
      avoid=$(e avoid retrieved)
      [[ -f "$avoid" ]] || { echo "skipping $tag: no $avoid" >&2; continue; }
      if [[ "$BENCH" == "sparc" ]]; then
        base="results/json/reflexion_root_${tag}_test.json"
        memp="results/json/memp_eval_retrieved_test_${tag}_v2.json"
      else
        base="results/json/reflexion_root_stepgame_${tag}_test${PTAG}.json"
        memp="results/json/memp_eval_stepgame_retrieved_test_${tag}${PTAG}.json"
      fi
      [[ -f "$base" ]] || { echo "skipping $tag: no baseline $base" >&2; continue; }
      echo "=== contrastive $BENCH $tag ==="
      python scripts/run.py contrastive --phase compare \
        --baseline "$base" --retrieved "$avoid" \
        $([[ -f "$(e hindsight retrieved)" ]]     && echo --hindsight "$(e hindsight retrieved)") \
        $([[ -f "$(e positive-only retrieved)" ]] && echo --positive  "$(e positive-only retrieved)") \
        $([[ -f "$(e avoid random)" ]]            && echo --random    "$(e avoid random)") \
        $([[ -f "$memp" ]]                        && echo --memp      "$memp") \
        --out "results/txt/contrastive_vs_baseline_${BM}${tag}${PTAG}.txt"
      # `balanced` differs from `avoid` in retrieval policy, not content, so it is compared separately.
      if [[ -f "$(e avoid balanced)" ]]; then
        echo "=== contrastive $BENCH $tag: balanced retrieval ==="
        python scripts/run.py contrastive --phase compare \
          --baseline "$base" --retrieved "$(e avoid balanced)" \
          $([[ -f "$(e hindsight retrieved)" ]] && echo --hindsight "$(e hindsight retrieved)") \
          --out "results/txt/contrastive_balanced_${BM}${tag}${PTAG}.txt"
      fi
    done
    ;;

  status)
    printf '%-14s %-22s %s\n' BENCH ARTEFACT STATE
    for b in sparc stepgame; do
      bm=""; pt=""
      [[ "$b" == stepgame ]] && { bm="stepgame_"; pt="_upstream-header"; }
      for spec in "${ALL_MODELS[@]}"; do
        IFS='|' read -r tag _r <<< "$spec"
        f="results/json/contrastive_bank_${bm}${tag}${pt}.json"
        [[ -f "$f" ]] && printf '%-14s %-22s %s\n' "$b" "bank $tag" "$(du -h "$f" | cut -f1)"
      done
      for f in results/json/contrastive_eval_${bm}*.json; do
        [[ -e "$f" ]] || continue
        n=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));s=d.get('summary',{});print(f\"{s.get('success','?')}/{s.get('n','?')} solved\" if s else f\"{len(d.get('puzzles',{}))} done (running)\")" "$f" 2>/dev/null || echo '?')
        printf '%-14s %-22s %s\n' "$b" "$(basename "$f" .json | sed "s/contrastive_eval_${bm}//;s/_test_/ /")" "$n"
      done
    done
    ;;

  *)
    awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 1 ;;
esac
