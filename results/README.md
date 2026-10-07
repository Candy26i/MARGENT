# Saved results

Two record sets are released with this snapshot. Neither is the paper's main
Qwen3.5-9B evaluation (Tables 1–5); those per-example branch and evaluation
records are exported separately with the paper's dashboard.

## `predecessor_scaling/main_results.json` — paper Appendix B (Table 6)

The predecessor diagnostic run before the marginal-value experiments: accuracy
on MedQA (n = 100), LegalBench (n = 200), GPQA (n = 100) and MMLU-Pro (n = 200)
development sets for Qwen3-4B, Qwen3-8B and Qwen3.5-9B managers, each as
direct-answer baseline, cold-start SFT, and outcome-only GRPO with generic or
task-specific sub-agent variants at several GRPO group sizes (`4gen` / `6gen` /
`8gen` = GRPO group size). Only aggregate accuracies were kept.
Table 6 of the paper is the best variant per (task, size) minus the same-size
base model; LegalBench was dropped from the paper. The cold-start and GRPO
code that produced these numbers is on the `legacy` branch (tag `v0.1-full`),
not on the main branch.

```bash
python scripts/summarize_main_results.py
```

## `medqa_marginal_v1/` — paper Appendix D (Table 10, "MedQA, 8B manager")

An earlier Qwen3-8B run of the interventional collection followed by an
outcome-only GRPO continuation (terminal binary correctness, no call penalty),
produced with the GRPO continuation code on the `legacy` branch (tag
`v0.1-full`); the main branch no longer contains that trainer. It documents
the collapse to three calls per example; it is **not** the selective SFT
policy reported in the main text.

| File | Produced by | Contents |
|---|---|---|
| `marginal_value_report.json` | `build_marginal_sft` | summary of the interventional collection (400 training questions, depth 1) |
| `counterfactual_records.jsonl` | `build_marginal_sft` | per question: direct candidate, the three forced one-step branches with full trajectories, the selected shortest-success target |
| `eval_dev200.jsonl`, `eval_dev200_report.json` | `eval_manager_tools` | GRPO-continued manager on the first 200 questions of the official MedQA dev split (`example_id` 10178–10377) |
| `eval_test200.jsonl`, `eval_test200_report.json` | `eval_manager_tools` | the same manager on the first 200 questions of the official MedQA test split (`example_id` 11450–11649) |

Each evaluation row stores the initial candidate (`initial_draft`), every
sub-agent call and output, the final answer, and the `corrected_by_tools` /
`corrupted_by_tools` flags. `manager_dir` in the report files names the
checkpoint directory used during the original run (`grpo_binary_mv2`).

Known caveat: the 400 collection questions of this run overlap the sub-agent
SFT questions (the collection command lacked `--exclude_sft_example_ids`), so
its one-step rescue rates are an optimistic estimate. The dev/test evaluations
do not overlap the SFT questions.

Recompute every number (Wilson 95% intervals, exact McNemar tests):

```bash
python scripts/analyze_results.py results/medqa_marginal_v1
```
