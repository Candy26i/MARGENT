# Marginal-value distillation: experiment protocol

"Advisor" in this document and in the report keys (`by_advisor_one_step`,
`available_advisors`) = sub-agent in the paper.

This is the protocol for the paper's method. The manager is trained by
supervised distillation only: `build_marginal_sft` collects same-state
counterfactual branches, `train_manager_sft` fine-tunes the manager on the
selected decisions, and `eval_manager_tools` / `eval_manager_forced` measure
the result. There is no reinforcement-learning step. The outcome-only GRPO
continuation discussed in the paper's Appendix D lives on the `legacy` branch
(tag `v0.1-full`) and is not part of this protocol.

The central training signal is built from one model-generated
`DRAFT_ANSWER`: training-time interventions compare `COMMIT` with each advisor
call. Ground truth selects a shortest trajectory that actually becomes
correct. Incorrect trajectories are never preferred merely because they use
fewer calls.

## 1. What `build_marginal_sft` produces

`build_marginal_sft` writes four artifacts under
`outputs/manager/$TEACHER_ID/marginal_value/` (or `--mv_output_dir`):

- `counterfactual_records.jsonl`: one record per question, containing the
  direct draft, all evaluated branches, and the preferred shortest sequence;
- `counterfactual_branches.jsonl`: one row per forced advisor sequence;
- `manager_sft_marginal.jsonl`: per-turn manager SFT data;
- `marginal_value_report.json`: direct accuracy, oracle accuracy/gain,
  per-advisor rescue/corruption rates (`by_advisor_one_step`), selected
  depths (`preferred_depth_counts`), and the first-tool distribution.

The selection rule is lexicographic:

1. correct beats incorrect;
2. if direct and advisor-assisted answers are both correct, commit;
3. if direct is wrong and an advisor path is correct, call a shortest such path;
4. if no evaluated path is correct, emit no routing SFT target for that question.

## 2. Fixed experimental setup

Run all commands from the repository root.

```bash
export BASE_MODEL="Qwen/Qwen3.5-9B"   # the paper's manager checkpoint; Qwen/Qwen3-8B also runs unchanged
export TEACHER_ID="medqa_marginal_v1"
export MEDQA_CACHE="outputs/data/medqa_us4_normalized.jsonl"
export TASK_DESC="You are a manager agent solving a medical multiple-choice question."
export TRAIN_SIZE=1400
export DEV_SIZE=200
export TEST_SIZE=500
export SUBAGENT_SERVER_URL="http://localhost:8000"
```

Use the same data split, seed, advisor checkpoints, manager base model, and
generation limits for every main comparison. Do not choose thresholds or
checkpoints using the test split.

Before starting, verify that these exist:

```bash
for KIND in extractor reasoner verifier; do
  test -d "outputs/adapters/${SUBAGENT_TEACHER_ID:-$TEACHER_ID}/${KIND}_adapter" || echo "missing $KIND"
done
```

If advisors are stored under a different run namespace, add
`--subagent_teacher_id "$SUBAGENT_TEACHER_ID"` to every command.

**Keep counterfactual questions disjoint from advisor SFT questions.** The
advisors are fine-tuned on teacher outputs for part of the same training
window, so pass their SFT files to every `build_marginal_sft` call:

```bash
SFT_DIR="outputs/sft_data/${SUBAGENT_TEACHER_ID:-$TEACHER_ID}"
export EXCL_ADVISOR_SFT="--exclude_sft_example_ids $SFT_DIR/extractor_sft.jsonl \
  --exclude_sft_example_ids $SFT_DIR/reasoner_sft.jsonl \
  --exclude_sft_example_ids $SFT_DIR/verifier_sft.jsonl"
```

(The offline `import_deepseek_jsonl` path writes the same `<kind>_sft.jsonl`
files unless `--deepseek_sft_jsonl` named another path; pass whichever path was
used.) Without this flag the counterfactual rescue rates are measured partly on
questions the advisors were trained on.

Start the LoRA advisor server on GPU 0 before counterfactual collection and
keep the collector/manager on another GPU:

```bash
# terminal A
bash scripts/start_subagent_server.sh "$BASE_MODEL" "${SUBAGENT_TEACHER_ID:-$TEACHER_ID}"
```

In terminal B, pin single-manager collection and evaluation to another GPU:

```bash
export CUDA_VISIBLE_DEVICES=1
```

The commands below use `--subagent_server_url`. Omit that flag only for a
small-model smoke test where all three local advisor instances fit in memory.

## 3. Step 0: advisor validity and direct baseline

First evaluate advisor schema validity. A malformed advisor cannot have
learnable marginal value.

```bash
python -m src.pipeline.cli eval_subagents \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  --eval_n_samples 100
```

Record `json_ok_rate` and `schema_ok_rate` for every advisor
(`outputs/eval/$TEACHER_ID/subagent_eval_report.json`). Do not continue if
schema validity is below 95%; inspect advisor prompts/checkpoints first.

The manager's direct baseline needs no separate stage. On the collection pool
it is `direct_accuracy` in `marginal_value_report.json` (Step 1): the accuracy
of the temperature-0 candidate every branch starts from, written by
`build_marginal_sft`. On dev and test it is `initial_draft_accuracy` in the
`eval_manager_tools` report (Step 4): the accuracy of the candidate the
manager states before any delegation, measured on the same questions as the
policy. The `none` sequence of Step 5 is the zero-delegation run of whichever
manager `--eval_manager_dir` names.

## 4. Step 1: cheap counterfactual smoke test

Start with 24 questions and one-step branches. This checks model loading,
native tool messages, answer parsing, and whether at least one advisor can
rescue a direct error.

```bash
python -m src.pipeline.cli build_marginal_sft \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  $EXCL_ADVISOR_SFT \
  --mv_manager_dir "$BASE_MODEL" \
  --mv_n_samples 24 \
  --mv_max_depth 1 \
  --mv_max_commit_rescue_ratio 1.0 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

Inspect:

```bash
python -m json.tool \
  "outputs/manager/$TEACHER_ID/marginal_value/marginal_value_report.json"
```

Smoke-test gates:

- `direct_valid_rate >= 0.95`;
- all three advisors have `n > 0` under `by_advisor_one_step`;
- `n_sft_turns > 0`;
- inspect at least ten `counterfactual_records.jsonl` rows manually;
- a rescued row must contain a wrong model draft, an actual tool output, and a
  correct model revision. It must not contain a GT answer inserted as a draft.

## 5. Step 2: full collection and the depth sweep

Run the one-step collection on 300–500 training questions. Use deterministic
manager generation (`--mv_temperature 0`) so paired differences are
attributable to the forced advisor rather than sampling noise.

```bash
python -m src.pipeline.cli build_marginal_sft \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  $EXCL_ADVISOR_SFT \
  --mv_manager_dir "$BASE_MODEL" \
  --mv_n_samples 400 \
  --mv_max_depth 1 \
  --mv_max_commit_rescue_ratio 1.0 \
  --mv_temperature 0 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

Oracle-gain gate (`marginal_value_report.json`):

- `oracle_gain >= 0.03`: proceed to distillation;
- `0.01 <= oracle_gain < 0.03`: proceed as a pilot, but advisor usefulness is
  likely the main bottleneck;
- `oracle_gain < 0.01`: stop manager work and improve advisors. A routing
  policy cannot learn useful calls if forced advisors almost never repair a
  direct error;
- for each advisor, inspect `rescue_rate`, `corruption_rate`, and
  `net_marginal_rate`. A negative net advisor should not be made mandatory.

The threshold is a practical go/no-go rule, not a reported statistical claim.
Report confidence intervals in the paper.

**Depth sweep (1 / 2 / 3).** The collector evaluates every one-step branch,
then expands only direct-wrong questions with no one-step success to ordered
sequences of two, then three, distinct advisors, stopping at the first depth
with a success. Directly correct roots are expanded one step only, for
corruption analysis. Collect depth 2 and depth 3 into their own directories:

```bash
for DEPTH in 2 3; do
  python -m src.pipeline.cli build_marginal_sft \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" \
    --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
    $EXCL_ADVISOR_SFT \
    --mv_manager_dir "$BASE_MODEL" \
    --mv_n_samples 400 \
    --mv_max_depth "$DEPTH" \
    --mv_max_commit_rescue_ratio 1.0 \
    --mv_temperature 0 \
    --mv_output_dir "outputs/manager/$TEACHER_ID/marginal_value_d$DEPTH" \
    --subagent_server_url "$SUBAGENT_SERVER_URL" \
    --task_description "$TASK_DESC"
done
```

Depth gate: compare `oracle_gain` and `preferred_depth_counts` across the
three reports. Use depth 2 only if it adds a meaningful oracle gain over depth
1, and depth 3 only if it adds over depth 2. Do not use a deeper search merely
to create longer demonstrations. The paper searches to depth 3 on MedQA and
AQuA-RAT and to depth 2 on MMLU-Pro and GPQA.

## 6. Step 3: marginal-value distillation and the ρ sweep

Train the routing policy from the selected counterfactual decisions. Use the
collection directory of the depth that passed the previous gate.

```bash
export MV_DIR="outputs/manager/$TEACHER_ID/marginal_value"   # or marginal_value_d2 / _d3
export MV_SFT="$MV_DIR/manager_sft_marginal.jsonl"
export MV_ADAPTER="outputs/manager/$TEACHER_ID/sft_marginal"

python -m src.pipeline.cli train_manager_sft \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --manager_sft_train_jsonl "$MV_SFT" \
  --manager_sft_output_dir "$MV_ADAPTER" \
  --manager_sft_epochs 1 \
  --manager_sft_lr 1e-5 \
  --sft_max_seq_len 4096 \
  --sft_bs 1 --sft_grad_accum 8
```

The `--sft_*` batch/sequence flags are shared with `train_subagent`; pass
`--sft_no_lora` for full-parameter fine-tuning instead of a LoRA adapter.

**Commit-to-rescue ratio ρ (`--mv_max_commit_rescue_ratio`).** Commit
trajectories are capped at ρ times the rescue trajectories at the question
level: `0` keeps rescues only, `-1` keeps every direct-correct commit, and the
default `1.0` is balanced. Sweep ρ over 0 / 0.5 / 1 / 2 / 3 / −1, writing each
collection and each adapter to its own directory:

```bash
for RHO in 0 0.5 1 2 3 -1; do
  TAG="rho${RHO}"
  python -m src.pipeline.cli build_marginal_sft \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" \
    --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
    $EXCL_ADVISOR_SFT \
    --mv_manager_dir "$BASE_MODEL" \
    --mv_n_samples 400 \
    --mv_max_depth 3 \
    --mv_max_commit_rescue_ratio "$RHO" \
    --mv_temperature 0 \
    --mv_output_dir "outputs/manager/$TEACHER_ID/marginal_value_$TAG" \
    --subagent_server_url "$SUBAGENT_SERVER_URL" \
    --task_description "$TASK_DESC"
  python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --manager_sft_train_jsonl "outputs/manager/$TEACHER_ID/marginal_value_$TAG/manager_sft_marginal.jsonl" \
    --manager_sft_output_dir "outputs/manager/$TEACHER_ID/sft_marginal_$TAG" \
    --manager_sft_epochs 1 \
    --manager_sft_lr 1e-5 \
    --sft_max_seq_len 4096 \
    --sft_bs 1 --sft_grad_accum 8
done
```

Larger ρ lowers the call rate and usually raises the call gap. Choose ρ,
depth and the checkpoint on dev only (Step 4). `eval_manager_tools` (Step 4)
overwrites `manager_tool_eval.jsonl` and its report on every run, so copy the
report before evaluating the next variant of the sweep.

A second collection round from the distilled policy is optional: rerun
`build_marginal_sft` with `--mv_manager_dir "$MV_ADAPTER"` and continue
training with `--manager_sft_init_adapter "$MV_ADAPTER"` (without it
`train_manager_sft` restarts from `--base_model`). It is selected on dev like
any other variant.

## 7. Step 4: development-set evaluation gates

```bash
python -m src.pipeline.cli eval_manager_tools \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size 0 \
  --eval_n_samples "$DEV_SIZE" \
  --eval_manager_dir "$MV_ADAPTER" \
  --eval_max_tool_calls 3 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

The report (`outputs/eval/$TEACHER_ID/manager_tool_eval_report.json`)
contains:

- `accuracy`, `initial_draft_accuracy` (the "candidate" column) and
  `avg_tool_calls`;
- `tool_call_rate`, `call_rate_given_draft_wrong`,
  `call_rate_given_draft_correct` and `draft_conditioned_call_gap`;
- `correction_rate` and `corruption_rate`.

Required behavioral gate:

- tool call rate is neither 0 nor 1;
- `draft_conditioned_call_gap > 0`;
- correction rate exceeds corruption rate;
- accuracy is not materially below the direct baseline.

If this gate fails, do not proceed to the locked evaluation. Inspect the
counterfactual records, change the commit/rescue ratio or depth, or improve
advisor quality.

Compare the variants on dev:

| Model | Accuracy | Avg calls | Call rate | Call given draft wrong | Call given draft correct | Correction | Corruption |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct base | | 0 | 0 | 0 | 0 | 0 | 0 |
| Marginal SFT (ρ, depth) | | | | | | | |

Choose ρ, depth and the checkpoint only from dev behavior. Freeze all choices
before the test run.

## 8. Step 5: forced-sequence baselines

Fixed delegation sequences with the same manager and the same advisor pool
isolate the value of the learned stopping decision. `none` is the
zero-delegation baseline, `verifier` the always-Verifier baseline, and the
three-advisor sequence the force-all baseline; any subset gives a fixed-k
baseline, and the per-question best over all subsets is the stopping oracle.

```bash
for SEQ in none verifier "extractor,reasoner,verifier"; do
  python -m src.pipeline.cli eval_manager_forced \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" \
    --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size 0 \
    --eval_n_samples "$DEV_SIZE" \
    --eval_manager_dir "$MV_ADAPTER" \
    --eval_forced_tools "$SEQ" \
    --subagent_server_url "$SUBAGENT_SERVER_URL" \
    --task_description "$TASK_DESC"
done
```

Outputs go to `outputs/eval/$TEACHER_ID/manager_forced_<sequence>.jsonl` and
`manager_forced_<sequence>_report.json`; `--eval_out_tag` renames them.

The same sequences with the untrained base manager (`--eval_manager_dir
"$BASE_MODEL"`: a Hugging Face id or a full checkpoint directory loads without
an adapter) give the "base manager + forced advisors" baselines of Section 10;
tag the outputs so they do not overwrite the trained manager's:

```bash
for SEQ in verifier "extractor,reasoner,verifier"; do
  python -m src.pipeline.cli eval_manager_forced \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" \
    --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size 0 \
    --eval_n_samples "$DEV_SIZE" \
    --eval_manager_dir "$BASE_MODEL" \
    --eval_forced_tools "$SEQ" \
    --eval_out_tag "base_${SEQ//,/_}" \
    --subagent_server_url "$SUBAGENT_SERVER_URL" \
    --task_description "$TASK_DESC"
done
```

## 9. Step 6: locked evaluation and the other benchmarks

After freezing the selected manager, run exactly once on the MedQA test split.

```bash
export FINAL_MANAGER="$MV_ADAPTER"   # the dev-selected adapter

python -m src.pipeline.cli eval_manager_tools \
  --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  --eval_n_samples "$TEST_SIZE" \
  --eval_manager_dir "$FINAL_MANAGER" \
  --eval_max_tool_calls 3 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

Run the same forced sequences (Step 5) on the locked split with the same
`$FINAL_MANAGER`.

MMLU-Pro, GPQA and AQuA-RAT each get their own collection and manager:
repeat Steps 1–5 with that benchmark's cache flag
(`--mmlu_pro_normalized_cache`, `--gpqa_normalized_cache`,
`--aqua_rat_normalized_cache`), a matching `--task_description` and the
paper's depth (3 on AQuA-RAT, 2 on MMLU-Pro and GPQA; GPQA collects 200
states), then evaluate once on a set disjoint from the collection pool. An
evaluation-only cache is loaded with `--train_size 0`, which makes the loader
honor the benchmark's own split labels. AQuA-RAT (collection pool from the
train split; the paper's locked set is the 254-question test split):

```bash
# collection pool: a capped slice of the 97k-question train split
python -m src.pipeline.cli load_aqua_rat \
  --aqua_rat_splits train --aqua_rat_max 2000 \
  --aqua_rat_normalized_cache outputs/data/aqua_rat_train.jsonl
# locked evaluation set
python -m src.pipeline.cli load_aqua_rat --train_size 0 \
  --aqua_rat_normalized_cache outputs/data/aqua_rat_test.jsonl

# Steps 1-5 with --teacher_id aqua_rat_mv, the train cache, --mv_max_depth 3
# and the task description below; then the single locked run:
python -m src.pipeline.cli eval_manager_tools \
  --base_model "$BASE_MODEL" --teacher_id aqua_rat_mv \
  --aqua_rat_normalized_cache outputs/data/aqua_rat_test.jsonl \
  --train_size 0 --dev_size 0 --test_size 254 \
  --eval_n_samples 254 \
  --eval_manager_dir "outputs/manager/aqua_rat_mv/sft_marginal" \
  --eval_max_tool_calls 3 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "You are a manager agent solving a multiple-choice algebra word problem."
```

For GPQA build the held-out Diamond cache with `scripts/build_gpqa_splits.py`
(see README); for MMLU-Pro load `--mmlu_pro_splits test`.

**Do not retune on the locked set.** ρ, depth, the checkpoint, the
`--eval_max_tool_calls` budget and any threshold are fixed on dev before the
single test run of each benchmark.

## 10. Required baselines and ablations

Main baselines:

1. base manager, direct answer: the candidate accuracy, `initial_draft_accuracy`
   in the `eval_manager_tools` report and `direct_accuracy` in the collection
   report (Step 0);
2. base manager + all advisors forced (`eval_manager_forced`, three-advisor
   sequence);
3. always-Verifier (`eval_manager_forced --eval_forced_tools verifier`);
4. random advisor routing with matched average calls (mix the `none` and
   fixed-k forced rows per question at the policy's call rate);
5. marginal-value SFT (main method, `eval_manager_tools`).

Core ablations:

1. replace the real initial draft with GT (expected to damage routing);
2. select advisor sequences without checking the counterfactual outcome;
3. remove the correct-correct commit tie-break;
4. remove commit/rescue balancing (`--mv_max_commit_rescue_ratio -1`);
5. depth 1 versus depth 2 versus depth 3 marginal search;
6. omit `DRAFT_ANSWER` from routing turns.

## 11. Statistical reporting

- Use at least three random seeds for the main method and strongest baselines.
- Report mean and standard deviation for accuracy and average calls.
- Use paired bootstrap confidence intervals on per-question accuracy
  differences because systems are evaluated on the same questions.
- Bootstrap average-call differences and correction/corruption differences.
- Report oracle gain with a confidence interval; it defines the maximum
  exploitable value of the current advisors.
- Include the full call-count distribution (`k=0,1,2,3`), not only its mean.

The paper's stopping claim should be supported by both outcomes and behavior:
high `call_rate_given_draft_wrong`, low `call_rate_given_draft_correct`, positive
correction-minus-corruption, and competitive final accuracy at fewer calls.

## 12. Failure diagnosis order

When a run fails, inspect in this order:

1. **No oracle gain:** advisors do not repair drafts; fix advisors.
2. **Oracle gain but empty SFT:** parsing or branch materialization bug.
3. **SFT call rate is zero:** commit examples dominate or tool calls do not
   render correctly in the tokenizer chat template; lower ρ.
4. **SFT calls everything:** reduce rescue oversampling (raise ρ) and check
   whether direct-correct commit rows are present.
5. **Calls occur but do not correct:** advisor evidence is not being integrated;
   inspect manager revisions and add those fresh failures to a second
   collection round from the distilled policy.
6. **Accuracy rises but calls also rise:** verify that the correct-correct commit
   tie-break is present and collect fresh marginal pairs from the new policy.

This order separates advisor capability, counterfactual data quality and
routing-policy distillation instead of trying to repair all three at once.
