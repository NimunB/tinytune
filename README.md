# tinytune

A small library for post-training small language models: supervised fine-tuning, DPO and GRPO, with pluggable rewards and a sandbox for running model-written code. It is the shared base for `strudel-gen`, `shader-gen`, `legal-bill-explainer` and `drawing2brep`.

> **Status:** in development. This README describes the planned design. Results are added as they are measured.

## What it does

1. Generates training data with a teacher model and keeps only examples that pass a reward threshold.
2. Runs SFT, DPO and GRPO from a YAML config.
3. Scores model outputs with a reward you define, executing any generated code in a sandbox.
4. Saves every rollout with its per-component reward.
5. Produces a results table, including a best-of-N baseline.

## Design

- **Thin on optimisers.** SFT, DPO and GRPO wrap the trainers in Hugging Face TRL. The library's own work is everything around them.
- **Reward interface.** A hard gate multiplied by a weighted sum of named components. Each component is logged separately, so when a model games one part of the reward it shows up in that part's curve.
- **Sandbox.** Model-written code runs in Docker containers with no network, a read-only filesystem, and CPU, memory and time limits.
- **Saved rollouts.** Prompt, completion and per-component reward for every sample, for later analysis.
- **Best-of-N in the eval runner.** Sampling N outputs and keeping the best by reward is the inference-time alternative to RL, so every project reports it next to GRPO.

## Evaluation

Toy task: emit JSON that matches a sampled spec.

| Model | Valid JSON | Matches spec |
| --- | --- | --- |
| Base | | |
| SFT | | |
| GRPO | | |
| Best-of-N | | |

Sandbox: containment tests (infinite loop, fork bomb, network call, file write) and executions per second.

Multi-GPU: throughput of one full fine-tune with FSDP at 1, 2 and 4 GPUs.

## Roadmap

MVP

- [ ] Config schema and reward interface
- [ ] Sandbox, one container per call
- [ ] Teacher data script with reward filter
- [ ] SFT and GRPO wrappers
- [ ] Eval runner with best-of-N
- [ ] Toy task end to end

After the MVP

- [ ] Warm container pool with throughput numbers
- [ ] DPO pair builder and wrapper
- [ ] FSDP run across 2 to 4 GPUs and a SLURM script
- [ ] Own GRPO loop, checked against TRL

## Quickstart

To be added with the first working version.
