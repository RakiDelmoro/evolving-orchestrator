# Agent Guidelines

These rules govern every change to Evolving Orchestrator. They are not suggestions.

## Vocabulary

| Name | What it is |
|---|---|
| **Design** | Opaque text between two markers: roles, instructions, contracts, call order. |
| **Episode** | One execution of a design in one workspace. |
| **Trajectory** | Messages, reasoning, tool calls, and tool results from an episode. |
| **Judge** | Pairwise workspace comparison returning `A`, `B`, or `tie`. |
| **Champion** | Best design found so far for a task. |
| **Generation** | One search → collect → rank → train → export cycle. |

Use these names. A “task” is one prompt; an “episode” is one run.

## Stack

- Rust 2021. The crate is `evolve` in `crates/evolving-orchestrator`.
- `#![forbid(unsafe_code)]` in the library and the binary.
- Dependencies are deliberate: `clap`, `serde`, `serde_json`, `thiserror`,
  `ureq`, and `glob`. Do not add one without a concrete need and a note
  explaining what it replaces.
- External effects live behind ports: executor, chat client, tokenizer,
  trainer, exporter. Pure decisions stay in ordinary functions.

## Sequential invariant

The pipeline is sequential. One episode runs at a time, one inference server
serves it, and the next stage starts only after the previous one finishes.

- Do not add `max_concurrent`, `parallel`, worker-pool, or background-agent
  settings.
- Collection is a loop, not a pool.
- Sub-agent calls are foreground and wait for the child.
- If a future change needs concurrency, it is a separate design decision with a
  measured memory budget, not a hidden flag.

## Quality contract

Software tasks are accepted only when:

```text
all hard gates pass
AND every scored dimension >= its floor
AND weighted score >= the task threshold
```

- Hard gates: acceptance, build, tests, test integrity, lint, format, secrets,
  unsafe policy, dependency risk, license, debug artifacts, scope.
- Scored dimensions: correctness depth, architecture, security, readability,
  cleanliness, performance, scalability, reliability, test quality,
  documentation, privacy, observability, compatibility, operability,
  accessibility, reproducibility.
- A missing gate result is a failure, never a pass by omission.
- The model may not mark its own gate as passed. Gate results come from
  commands.
- A candidate that fails a gate cannot become champion or be published, no
  matter how good its score.

## Data crossing the boundary

Untrusted: task files, designs, model output, HTTP bodies, judge JSON, session
files, configuration. Validate at the boundary and include the offending path or
field in the error. Unknown configuration fields are rejected.

## Errors

- Every fallible operation returns `Result<T, EvolveError>`.
- Use the existing variants. Do not add a catch-all.
- Do not silently drop a task, episode, gate, or selected window.
- Preserve logs and evidence on failure; stop the stage with a precise reason.

## Testing

- Unit tests are in-memory and fast.
- Pure logic (config, markers, champion, schedule, Bradley–Terry, split,
  checkpoint selection, masks) is tested directly.
- External effects are faked behind ports.
- Golden fixtures cover deterministic artifacts: prompts, pair sets, scores,
  token ids, and labels.
- `cargo test --workspace` must pass.
- Do not claim a stage works without an integration test or a recorded run.

## Execution and training

- Pi is the executor; the prompt is passed as the final argument.
- One model, one endpoint, one server.
- Stop the inference server before training; restart it after export.
- Training uses QLoRA with FP16 compute on this GPU. Do not introduce BF16 or
  FP8 assumptions.
- Do not run training and inference at the same time on the 8 GB card.

## Layout

```text
crates/evolving-orchestrator/src/
  config.rs     schema + validation
  design.rs     markers + task catalog
  chat.rs       model client
  judge.rs      evidence + verdict
  exec.rs       sequential process/Pi runner
  pi_session.rs Pi JSONL parser
  io.rs         JSON/JSONL
  error.rs      error type
  main.rs       CLI
  history.rs    comparison records
  champion.rs   champion derivation
  prompt.rs     optimizer prompt
  search.rs     inner loop
  ranking.rs    schedule + Bradley–Terry
  select.rs     train/validation split
  render.rs     masks + windows
  train.rs      QLoRA command + checkpoint
  export.rs     merge + GGUF
  cycle.rs      stage orchestration
```

The Pi `agent` tool lives in `crates/evolving-orchestrator/pi/subagent.ts`.

The full specification is [`BLUEPRINT.md`](crates/evolving-orchestrator/BLUEPRINT.md).
