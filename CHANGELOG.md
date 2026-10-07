# Changelog

## v0.2 — simplified to the paper's method

The main branch now contains only what the paper uses: sub-agent synthesis
and LoRA SFT, the same-state interventional collection (`build_marginal_sft`),
marginal-value distillation (`train_manager_sft`), the three evaluations
(`eval_manager`, `eval_manager_tools`, `eval_manager_forced`) and the data
loaders. Everything removed below stays on the `legacy` branch (tag
`v0.1-full`) and is not maintained.

Removed (legacy branch only):

- stages `train_manager_grpo`, `evolve_build_sft`, `evolve_round`,
  `export_manager_coldstart_prompts`, `import_manager_coldstart_responses`,
  `manager_coldstart_sft`, `export_legalbench_jsonl` and the flags only they
  used (`--mgr_*` including the SFT anchor, `--coldstart_*`, `--wandb_*`,
  `--legalbench_*`, the evolve-only flags);
- `src/manager/grpo_train.py`, `reward.py` (ADC/CCR), `routing_anchor.py`,
  `evolve.py`, `src/benchmarks/legalbench.py`, `configs/` (accelerate /
  DeepSpeed), `scripts/train_manager_grpo_multigpu.sh`,
  `scripts/summarize_routing_trace.py`, `scripts/validate_sft_anchor.py`,
  `docs/history/`, `tests/test_routing_anchor.py`.

Moved, behaviour unchanged:

- `ManagerSFTConfig` / `train_manager_sft` from `evolve.py` to
  `src/manager/sft.py`;
- `src/pipeline/stages.py` split into `context.py`, `data.py`,
  `subagent_stages.py`, `manager_stages.py` and `eval_stages.py`;
- the three copies of the tool-call-argument normaliser and chat renderer
  replaced by `src/manager/chat_template.py` (unit-tested).

Added:

- AQuA-RAT loader (`src/benchmarks/aqua_rat.py`, stage `load_aqua_rat`,
  `--aqua_rat_*` flags; the paper's test split, n = 254) with
  `tests/test_aqua_rat.py`;
- `tests/test_chat_template.py`; CI runs `unittest discover`;
- the CLI parser is grouped per stage family, so `--help` reads by stage;
- `src.utils` no longer imports torch at import time, so the unit tests run
  on a stdlib-only Python.

Docs: `README.md` and `docs/EXPERIMENTS.md` describe the SFT protocol only;
`results/README.md` points to the legacy branch for the Appendix B/D runs.

## v0.1-full (tag)

Initial release: the full research snapshot, including the outcome-only GRPO
continuation, the ADC/CCR rewards, the SFT anchor, the failure-recycling
evolve loop, the cold-start stages and LegalBench. Kept as the `legacy`
branch.
