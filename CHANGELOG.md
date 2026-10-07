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
  replaced by `src/manager/chat_template.py` (unit-tested), which keeps the
  `stages.py` / `marginal_value.py` behaviour (invalid or non-mapping
  tool-call arguments become `{}`); the `evolve.py` copy that
  `train_manager_sft` used had wrapped unparsable strings as
  `{"input": ...}`. Unreachable with pipeline-written JSONL, where every
  argument string is `json.dumps` of a dict;
- one manager loader (`src/manager/loading.py`), one sub-agent pool builder
  (`build_subagent_pool` in `src/subagents/runtime.py`), one
  `tool_call_message` and one `mask_prefix_len` (`chat_template.py`) replace
  the per-stage copies; the sub-agent SFT/runtime render through
  `chat_template.render_chat` (identical output: their messages carry no tool
  calls). Dead code dropped: `exploration_hint` / `_token_to_label` in
  `prompt.py` (the rendered system prompt is byte-identical).

Changed defaults:

- `train_manager_sft` reads `outputs/manager/<id>/marginal_value/manager_sft_marginal.jsonl`
  and writes `outputs/manager/<id>/sft_marginal/` (was
  `evolve/manager_sft_from_failures.jsonl` and `sft_evolved/`);
  `eval_manager`, `eval_manager_tools` and `eval_manager_forced` default
  `--eval_manager_dir` to that `sft_marginal/` directory (was the GRPO
  directory). Legacy `sft_evolved/` and `grpo/` checkpoints are not looked up;
  pass them explicitly.
- `--binding_mode auto` means the same thing on both sides: `build_marginal_sft`
  and `train_manager_sft` use environment binding, `train_manager_sft` saves it
  to `manager_run_config.json` next to the checkpoint (the removed GRPO trainer
  used to write that file, so nothing on v0.2 did), and the eval stages read it
  back, falling back to environment instead of argument when the file is
  absent (e.g. for the untrained base model).
- One tool-schema builder, `manager_tool_schemas` in `src/manager/prompt.py`,
  with the two wordings the pipeline always used: `descriptions="training"`
  (collection and SFT) and `descriptions="evaluation"` (the eval stages, so
  evaluations stay comparable with the released runs).
- `load_<benchmark>` stages always load the benchmark they name, even when
  another benchmark's cache flag is set; `load_manager` fails fast on a
  mistyped local checkpoint path.
- `--eval_manager_dir` accepts a Hugging Face model id or a full checkpoint
  directory in all three eval stages (`eval_manager_tools` /
  `eval_manager_forced` required a local directory and treated anything
  without `config.json` as an adapter).
- `requirements.txt`: `trl`, `wandb` and `pyyaml` dropped (nothing imports
  them), `requests` added (RemoteSubagentPool); `accelerate` stays
  (`transformers.Trainer` needs it).

Added:

- AQuA-RAT loader (`src/benchmarks/aqua_rat.py`, stage `load_aqua_rat`,
  `--aqua_rat_*` flags; the paper's test split, n = 254) with
  `tests/test_aqua_rat.py`;
- `tests/test_chat_template.py`; CI runs `unittest discover` and pyflakes
  (zero warnings);
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
