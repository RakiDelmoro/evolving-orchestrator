# Evolving Orchestrator

Evolving Orchestrator is a workflow-evolution harness for software work.

It treats the organization of a team of coding agents as data: a **design**
block that says who the agents are, what each one does, what each must return,
and in what order. Evolving Orchestrator runs a design, judges the resulting workspace, asks the
model to improve the design, and keeps the best one. Optionally, it ranks the
best design's episodes and fine-tunes the shared model on the selected
trajectories so the next generation is better on its own.

```text
L0 -- design search --> d* -- collection --> episodes -- rank --> SFT -- train --> L1
```

The inner loop changes only the design text. The outer loop changes the weights.
Both are optional: you can use Evolving Orchestrator in **task mode** and never train.

- **Task mode:** search for a good design, run it, publish the best workspace.
- **Model mode:** task mode plus render, train, and export a new model.

## Status

`evolve` 0.1.0 is a working scaffold:

- `config` — cycle schema and boundary validation.
- `design` — marker parsing and the task catalog.
- `chat` — OpenAI-compatible client.
- `judge` — workspace evidence and pairwise verdicts.
- `exec` — sequential process/Pi runner: prompt, workspace, timeout, status, sessions.
- `pi_session` — Pi JSONL parser for text, thinking, tool calls, and tool results.
- `history` / `champion` — comparison records and champion arbitration.
- `ranking` / `select` — CPython-compatible tournament, Bradley–Terry, and the split.
- `prompt` / `search` — optimizer prompt and the design search loop.
- `collect` — teacher episodes under the retained design.
- `render` — masked SFT windows.
- `train` / `export` — checkpoint selection and the merge/quantize command ports.
- `cycle` — all stages in sequence.
- `pi/subagent.ts` — Pi `agent` tool, one synchronous child, hard depth 1.
- `io` / `error` — file helpers and one error type.

Every stage is exposed by the CLI:

`validate → plan → search → collect → rank → select → render → train → export → cycle`

## Requirements

- Rust and Cargo (edition 2021).
- `git`.
- An OpenAI-compatible model server.
- For the executor: `pi` with a local provider, or any command that accepts a
  prompt.
- For training: Python, PyTorch, PEFT, and bitsandbytes on a CUDA GPU.

The pilot target is one 8 GB GPU and Qwen2.5-Coder-3B-Instruct Q4_K_M. The
pipeline is sequential by design: one episode at a time, one inference server.
There is no concurrency setting.

## Quick start

## 35B MoE (inference only)

Qwen3.5-35B-A3B Q4_K_M is 22.29 GB and fits in the host's 62 GB RAM. Serve it
with `-cmoe` so the MoE experts stay on CPU while attention and KV cache use
the 8 GB GPU:

```bash
llama-server -m ~/models/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --alias Qwen3.5-35B-A3B --jinja -ngl 999 -cmoe -c 65536 -t 12 \
  --host 127.0.0.1 --port 8080
```

Then run task mode, which stops before training:

```bash
evolve cycle --config configs/evolve/local-35b.json \
  --tasks configs/evolve/tasks --task-only
```

See [`crates/evolving-orchestrator/35B-MOE.md`](crates/evolving-orchestrator/35B-MOE.md)
for memory tuning and the Pi provider example.


```bash
# 1. Serve the local model (one model, one port).
llama-server \
  -m qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  --jinja -ngl 999 -c 8192 \
  --host 127.0.0.1 --port 8080

# 2. Check the cycle configuration and task files.
cargo run -p evolving-orchestrator -- validate \
  --config configs/evolve/local-3b.json \
  --tasks configs/evolve/tasks

# 3. Print the plan, seeds, and stage order.
cargo run -p evolving-orchestrator -- plan \
  --config configs/evolve/local-3b.json \
  --tasks configs/evolve/tasks
```

The pilot configuration uses one task, three design iterations, four collection
episodes, and a 100-step QLoRA run. It is meant to prove the loop, not to
reproduce a paper-scale result.

## Layout

```text
Cargo.toml                 workspace: only crates/evolving-orchestrator
configs/evolve/
  local-3b.json            pilot cycle configuration
  tasks/
    bare/query1.txt        task prompt
    initial/query1_team.txt task + initial design block
crates/evolving-orchestrator/
  BLUEPRINT.md             end-to-end specification
  src/
    config.rs              schema + validation
    design.rs              markers + task catalog
    chat.rs                model client
    judge.rs               evidence bundle + pairwise verdict
    io.rs                  JSON/JSONL helpers
    error.rs               error type
    main.rs                CLI
  pi/                      sub-agent extension (planned)
```

Run directories (`runs/`), model weights (`*.gguf`), and training output are
ignored by git.

## Quality contract

For software tasks, “solved” means more than passing tests. The blueprint
defines hard gates (acceptance, build, tests, test integrity, lint, format,
secrets, unsafe policy, dependency and license checks, debug artifacts, scope)
and scored dimensions (architecture, security, readability, cleanliness,
performance, scalability, reliability, test quality, documentation, privacy,
observability, compatibility, operability, accessibility). A candidate must
pass every gate, clear every dimension floor, and clear the weighted threshold.
Missing evidence is a failure.

## License

MIT. See [`LICENSE`](LICENSE).
