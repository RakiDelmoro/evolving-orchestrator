# Evolving Orchestrator — Implementation Blueprint

**Status:** final. Neutral naming only (`evolve`, `design`, `episode`, `judge`, `generation`).
**Rule:** the pipeline is sequential by construction. There is no concurrency setting and no parallelism flag. One episode runs at a time, one inference server serves it, and the next stage starts only after the previous one finishes.

---

## 1. Purpose

Evolving Orchestrator evolves a text **design** for a team of coding agents, ranks the resulting
episodes, and fine-tunes one shared model on the selected trajectories. The next
generation uses the updated model for every role.

```text
L0 -- design search --> d* -- collection --> episodes -- rank --> SFT -- train --> L1
```

The inner loop changes only the design text. The outer loop changes the weights.

Non-goals: 27B models, multi-GPU, FP8, vLLM, external judge APIs, and
reproducing published scores. The target is a working mechanism on one 8 GB GPU.

---

## 2. Hardware and model

Host: Quadro RTX 4000 **8 GB** (Turing, FP16 only), Xeon E-2176G, 62 GB RAM,
621 GB free.

| Workload | Qwen2.5-Coder-3B Q4 | Qwen2.5-Coder-7B Q4 |
|---|---:|---:|
| Weights in VRAM | ~2.0 GB | ~4.5 GB |
| One 8k conversation | ~0.3 GB | ~0.45 GB |
| One episode total | ~2.7 GB | ~5.5–6.0 GB |
| QLoRA at 1k sequence | ~4–5 GB | ~7.0–7.5 GB |

Pilot: **3B**. Move to 7B only after the full loop works. The model is shared by
the executor, judge, and optimizer. Training stops the inference server; export
restarts it with the new model.

---

## 3. Vocabulary

| Term | Meaning |
|---|---|
| Task | One prompt file. |
| Design | Opaque text between a start marker and an end marker. |
| Episode | One execution of a design in one workspace. |
| Trajectory | Messages, reasoning, tool calls, and tool results. |
| Judge | Pairwise workspace comparison returning `A`, `B`, or `tie`. |
| Optimizer | Model call that rewrites the design. |
| Champion | Best design so far for a task. |
| Baseline | Bare task with no design. |
| Retained design | Champion saved for collection. |
| Generation | One search → collect → rank → train → export cycle. |

---

## 4. Flow

```text
for generation g:
    for each training task:
        run bare baseline
        design = v0
        for k = 1..=N:
            run team(design) -> workspace
            verdict = judge(workspace, champion)
            if verdict wins: champion = design
            design = optimizer(task, design, champion, history, evidence)
        retained[task] = champion

    for each task with a retained design:
        run it M times -> episodes
        rank episodes pairwise -> Bradley–Terry
        ranks 1..K-1 -> train, rank K -> validation

    render conversations -> masked windows
    stop inference server
    QLoRA train -> adapter -> select checkpoint
    merge adapter -> FP16 model -> GGUF Q4_K_M
    restart inference server with the new model
```

---

## 5. Crate layout

```text
crates/evolving-orchestrator/
  Cargo.toml
  BLUEPRINT.md
  src/
    lib.rs
    main.rs          # CLI: validate, plan (live); other stages below
    config.rs        # schema + validation (live)
    design.rs        # markers + task catalog (live)
    io.rs            # JSON/JSONL helpers (live)
    error.rs         # error type (live)
    chat.rs          # OpenAI-compatible client (live)
    judge.rs         # evidence bundle + pairwise verdict (live)
    exec.rs          # EpisodeRunner
    pi_session.rs    # Pi JSONL -> messages/threads
    history.rs       # comparison records
    champion.rs      # champion derivation
    prompt.rs        # optimizer prompt + normalization
    search.rs        # inner loop
    ranking.rs       # CPython RNG, schedule, Bradley–Terry
    select.rs        # train/validation split
    render.rs        # masks + windows + tokenizer port
    train.rs         # QLoRA command + checkpoint selection
    export.rs        # merge + GGUF conversion
    cycle.rs         # stage orchestration
  pi/
    subagent.ts      # `agent` tool for Pi
  tests/
```

Ports:

```rust
trait EpisodeRunner { fn run(&self, spec: &EpisodeSpec) -> Result<EpisodeOutcome>; }
trait ChatClient    { fn complete(&self, system: &str, user: &str, opts: &ChatOptions) -> Result<String>; }
trait PairwiseJudge { fn compare(&self, task: &str, a: &Bundle, b: &Bundle, seed: u64) -> Result<Verdict>; }
trait Tokenizer     { fn encode(&self, text: &str) -> Result<Vec<u32>>; }
trait Trainer       { fn train(&self, request: &TrainRequest) -> Result<TrainOutcome>; }
trait Exporter      { fn merge_and_export(&self, request: &ExportRequest) -> Result<()>; }
```

Only the executor goes through Pi. Judge and optimizer call the inference
server directly.

---

## 6. Data layout

```text
runs/L0/
  config.json
  reference/query1/
  workspaces/v0/query1_team/
  workspaces/v1/query1_team/
  logs/reference/query1/
  logs/search/v0/query1_team/
  search/query1_team/
    versions/multi_agent_design_v0.txt
    versions/multi_agent_design_v1.txt
    updated_queries/query1_team-v1.txt
    comparisons/v0_vs_v1/result.json
    records/comparison_history.jsonl
    records/champion_history.jsonl
    retained_design.json
  pool/workspaces/t1_L0_s710101/
  pool/runlogs/t1_L0_s710101/
  pool.json
  selection/task1_matches.jsonl
  selection/task1_ranked.json
  selection/train.json
  selection/val.json
  data/train.jsonl
  data/val.jsonl
  training/train_log.jsonl
  training/step_00025/lora_state_dict.safetensors
  selected_checkpoint.json
  exported/bf16/
  exported/gguf/model-Q4_K_M.gguf
  outputs/
    query1/
      workspace/                  # copy of the best task result
      manifest.json               # design version, judge history, test log
```

Every episode keeps its own workspace under `workspaces/` or `pool/`. At the end
of search, the champion's workspace is copied to `outputs/query{id}/workspace/`
so the task result is a first-class deliverable, not a by-product.

Pretty JSON is two-space indented with a trailing newline. Records are one
compact object per line.

---

## 7. Configuration

Pilot: `configs/evolve/local-3b.json`

```json
{
  "generation": 0,
  "run_dir": "runs/L0",
  "model_id": "qwen2.5-coder-3b-instruct-q4_k_m",
  "endpoint": "http://127.0.0.1:8080/v1",
  "tasks": {
    "train_candidates": [1],
    "train": [1],
    "test": [],
    "excluded": []
  },
  "search": { "iterations": 3, "seed_base": 450000, "rule": "retained-best" },
  "collection": { "candidates": 4, "train_ranks": 3, "validation_rank": 4, "seed_base": 710000 },
  "ranking": { "opponents_per_episode": 2, "seed": 1234 },
  "training": {
    "window": 1024, "context_overlap": 128, "steps": 100,
    "lr": 0.0002, "warmup_steps": 5, "min_lr_ratio": 0.1,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "save_every": 25, "val_every": 50,
    "replicas": 1, "gpus_per_rank": 1, "grad_accum": 3,
    "ce_chunk": 128, "seed": 1234
  },
  "executor": {
    "program": "pi",
    "args": ["--print", "--mode", "json", "--provider", "local",
             "--model", "qwen2.5-coder-3b-instruct-q4_k_m",
             "--approve", "--no-context-files"],
    "stdin_prompt": false,
    "prompt_arg": true,
    "timeout_seconds": 1800
  },
  "judge": { "api_key_env": "EVOLVE_API_KEY", "temperature": 0.6,
             "top_p": 0.95, "max_output_tokens": 2048 }
}
```

Validation before any side effect:

- `train`, `test`, `excluded` are pairwise disjoint.
- `train ∪ excluded == train_candidates`.
- Endpoint is `http://127.0.0.1:PORT/v1`.
- `validation_rank == train_ranks + 1`.
- `candidates >= validation_rank`.
- `opponents_per_episode` is even and ≥ 2.
- `replicas × grad_accum == 3`.
- Every task has `bare/query{id}.txt` and `initial/query{id}_team.txt`.
- Exactly one of `stdin_prompt` and `prompt_arg` is true.

The scale recipe is numbers, not code:

```json
"search": {"iterations": 11, "seed_base": 450000}
"collection": {"candidates": 18, "train_ranks": 15, "validation_rank": 16, "seed_base": 710000}
"ranking": {"opponents_per_episode": 6, "seed": 1234}
"training": {"window": 49152, "context_overlap": 8192, "steps": 1624,
             "lr": 0.00003, "lora_r": 64, "lora_alpha": 128}
```

Do not use those on this GPU.

---

## 8. Stages

### 8.1 validate

Checks the config, task files, markers, and non-empty task text. Warns on an
unreachable endpoint; fails on a missing executor program. No side effects.

### 8.2 plan

Prints model, endpoint, generation, run directory, task split, seeds, training
hyperparameters, stage order, and the exact training/export commands. No side
effects.

### 8.3 search — inner loop

1. Write `v0` from the initial design.
2. Run the baseline once: bare task, condition `t{id}_bare`,
   seed `search.seed_base + id`.
3. For `k = 0..=iterations`:
   - Render prompt `v{k}` (initial file for `k=0`, updated query otherwise).
   - Run episode: condition `t{id}_v{k}`,
     seed `search.seed_base + id + k*1000`.
   - Compare:
     - `retained-best`: `v{k-1}` vs `v{k}`, key `v{k-1}_vs_v{k}`.
     - `fixed-reference`: `v{k}` vs the baseline, key `v{k}_vs_no_harness`.
   - Judge semantics: adjacent `A` = previous, `B` = current; baseline `A` =
     baseline, `B` = version.
   - Append a history row with key, winner, rationale, model, UTC time, repos.
   - Derive champion. Advance only on a direct win. If the current version won
     while an older champion reigns, run one arbitration comparison and append
     to `champion_history.jsonl`.
   - If `k < iterations`, propose `v{k+1}`:
     - Inputs: task, current design, champion design (if older), v0, history
       with diffs, workspace evidence, execution metadata.
     - Output: JSON with `improved_multi_agent_design`.
     - Require the start marker; trim at the end marker only near the tail.
     - Reject empty or malformed output and stop the task.
     - Write the next version and rendered query.
   - If `k == iterations`, evaluate only.
4. Write `retained_design.json` with `version` and `beats_reference`.
5. **Record the champion workspace** as the fallback task result. If collection
   runs later, the rank-1 collection workspace replaces it. This is the actual
   answer to the task; the design and the model are how it was obtained.

Failure policy: preserve logs; retry infrastructure failures once; stop the
task on model failure; never silently drop a task.

### 8.4 collect

Teacher prompt = retained design + `\n\n` + teacher directive.

For `i = 0..candidates`:

- seed = `collection.seed_base + 100*task + i`;
- condition = `t{task}_L{generation}`;
- name = `{condition}_s{seed}`;
- run, record in `pool.json`.

Failures are recorded, not fatal. Ranking sees only successful episodes with a
complete trajectory. Pilot: 4 candidates. Reference scale: 18.

### 8.5 rank

- Need at least `validation_rank` successful episodes.
- Sort names; shuffle with CPython-compatible `random.Random(seed + task)`;
  pair ring distances `1..opponents/2`; unique pairs; orient with
  `random.Random(seed + task + 1)`.
- Judge each pair with seed `ranking.seed + task*1000 + index`.
- Bradley–Terry: 500 iterations, prior 0.5, tie = half win, geometric-mean
  normalization, return `ln(score)`.
- Sort descending by score, then name. Write matches and ranked JSON.

The RNG is a faithful MT19937 + CPython `shuffle`/`random` port. Golden tests
cover seeds 1234 and 7.

### 8.6 select

- Ranks `1..train_ranks` → train; rank `validation_rank` → validation.
- Reject overlap, test tasks, and empty splits.
- Write `selection/train.json` and `selection/val.json`.
- **Publish the task result:** rank 1 is the best sample under the retained
  design. Copy its workspace to `outputs/query{id}/workspace/` and write
  `manifest.json` with the task id, source episode, design version, score,
  comparison history, and the test command/result. For a task without
  collection, publish the champion search workspace instead.

Pilot: 3 / 1. Reference scale: 15 / 1.

### 8.7 render

1. Parse Pi sessions: `system`, `user`, `assistant` (`text`, `thinking`,
   `toolCall`), `toolResult`.
2. Group threads: top-level session = orchestrator; `agent` child sessions =
   workers.
3. Substitute the teacher prompt with the bare task in the orchestrator's first
   user message. Do not alter worker messages.
4. Tokenize with the Qwen tokenizer and chat template.
   - Assistant reasoning, text, and tool calls: loss = 1.
   - System, user, tool results, and overlap context: label `-100`.
5. Window: header (system + first user) + body windows of `window` tokens with
   up to `context_overlap` masked context; include the segment query when
   needed; fail if a selected episode yields no windows.
6. Write `data/train.jsonl`, `data/val.jsonl`, and stats.

Pilot: 1024 / 128. Reference scale: 49152 / 8192. This is the parity-critical
stage; build golden token/label fixtures first.

### 8.8 train — the weight update

Preconditions: server stopped, VRAM free, both datasets non-empty.

Command adapter:

```bash
python train_qlora.py \
  --model <parent-fp16> \
  --train data/train.jsonl --val data/val.jsonl \
  --out training \
  --window 1024 --overlap 128 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --lr 2e-4 --warmup-steps 5 --min-lr-ratio 0.1 \
  --steps 100 --save-every 25 --val-every 50 \
  --grad-accum 3 --ce-chunk 128 --seed 1234
```

Behavior:

- 4-bit NF4 base, FP16 compute.
- LoRA on `q/k/v/o/gate/up/down`.
- Assistant-token loss only.
- Task-uniform, role-balanced sampling; orchestrator probability 1/3.
- AdamW `(0.9, 0.95)`, cosine decay.
- Save `step_000NN/lora_state_dict.safetensors`.
- Append `train_log.jsonl` with `step`, `val_loss`, `val_tokens`,
  `checkpoint`.

Checkpoint selection: ignore step 0; require finite loss and positive val
tokens; require the file; minimum loss wins; reject duplicate steps.

### 8.9 export

QLoRA saves **only the adapter**. The adapter is a pair of small matrices per
adapted layer, not a model. Deploying it requires applying:

```text
W_new = W_base + (alpha / r) * B @ A
```

That addition needs full-precision base weights. A Q4 GGUF stores quantized
integers plus block scales, not a dense `W`, so the adapter cannot be added to
it accurately. The FP16 directory below is therefore a **temporary build
artifact**, written on CPU RAM and deleted after quantization. It is not a
second training conversion. (Turing has no BF16; use FP16/FP32 on CPU.)

1. Merge the adapter into the full-precision base on CPU.
2. Write a complete FP16 HF model directory.
3. Convert and quantize:

```bash
python llama.cpp/convert_hf_to_gguf.py exported/bf16 --outfile exported/gguf/model-f16.gguf
llama-quantize exported/gguf/model-f16.gguf exported/gguf/model-Q4_K_M.gguf Q4_K_M
```

4. Start the server, check `/v1/models`, run one short completion.
5. Write `exported/READY`, then delete the temporary FP16 directory if disk
   space matters.

No FP8.

**Optional, not the default:** keep the adapter separate and serve a Q4 base
with a GGUF LoRA adapter (`llama-server --lora adapter.gguf`). This avoids the
merged file, but the adapter was trained against NF4 while the base is
Q4_K_M; quality can drift and every episode carries an extra runtime flag.
Use it only as an experiment.

### 8.10 cycle

```text
validate -> plan -> search -> collect -> rank -> select -> render
-> stop server -> train -> export -> restart server
```

Write a generation summary with model in/out, retained versions, episode count,
window count, train steps, and selected checkpoint. The next generation uses a
new `run_dir` and the exported model.

### 8.11 rollout / report (optional)

Bare-task rollouts and judge comparisons for measurement only. The training
loop never depends on them.

### 8.12 quality contract — what “solved” means

A candidate is **accepted** only when all three hold:

```text
all hard gates pass
AND every scored dimension >= its floor
AND weighted score >= the task threshold
```

A candidate that passes the tests but fails a gate is **rejected**, not
“mostly solved.” The report must say which gate failed and show the evidence.

#### Hard gates (fail closed)

| Gate | Check |
|---|---|
| Acceptance | The task’s explicit acceptance command passes. |
| Build | Compile / typecheck succeeds. |
| Tests | The project’s test command passes. |
| Test integrity | Test files are unchanged (hash before/after). |
| Lint | The project’s lint command passes. |
| Format | The project’s format check passes. |
| Secrets | No new secret-looking values in the diff. |
| Unsafe policy | No new `unsafe` / FFI / privileged call unless the task allows it. |
| Dependency risk | No new critical/high advisory; lockfile changes reviewed. |
| License | No new dependency outside the allowlist. |
| Debug artifacts | No new debug prints, commented-out blocks, or stray files. |
| Scope | Only files the task permits are modified. |

A missing gate result is a **failure**, never a pass by omission.

#### Scored dimensions

| Dimension | What it covers | Checked by |
|---|---|---|
| Correctness depth | Edge cases, error paths, invariants beyond the tests | Tests + review |
| Architecture | Boundaries, cohesion, coupling, dependency direction, testability, simplicity | Review |
| Security | Input validation, authn/authz, injection, crypto, least privilege, supply chain | Scanners + review |
| Readability | Naming, structure, comments that explain why, small cohesive units | Review + lint |
| Cleanliness | Dead code, duplication, formatting, file hygiene, no leftovers | Lint + review |
| Performance | Algorithmic complexity, allocations, measured benchmark vs budget | Bench + review |
| Scalability | Data structures, concurrency safety, backpressure, resource limits | Review + stress test |
| Reliability | Error handling, no panics/unwraps in production paths, timeouts, retries, graceful degradation | Review + tests |
| Test quality | Meaningful assertions, behavior coverage, determinism, no flaky or skipped tests | Coverage + review |
| Documentation | README, API docs, usage, design decisions, changelog | Review |
| Privacy | PII handling, logging, retention, data minimization | Review |
| Observability | Logs, metrics, traces, actionable errors | Review |
| Compatibility | API/ABI stability, config, migrations, platform support | Review + tests |
| Operability | Deploy, rollback, migrations, configuration | Review |
| Accessibility / UX | Only for user-facing work | Review |
| Reproducibility | Pinned deps, deterministic build, documented environment | Tooling + review |

#### Rules that keep the score honest

1. **Tests are a gate, not the score.** Passing tests cannot compensate for a
   security hole or an architectural mess.
2. **Deterministic tools first.** Build, tests, lint, format, secrets, audit,
   unsafe, coverage, and benchmarks are commands, not model opinions.
3. **The model may not mark its own gate as passed.** Gate results come from
   the commands; the model may only explain them.
4. **Security and architecture need evidence.** A review must cite file paths,
   line numbers, or command output. “Looks good” is not a score.
5. **Performance is a budget, not a goal.** If the task defines a budget, measure
   it. Otherwise judge algorithmic complexity and avoid rewarding
   micro-optimization that hurts readability.
6. **A low score on any dimension blocks publication** even when the weighted
   average is high. Floors are per dimension.
7. **No silent drops.** A rejected candidate is published as `REJECTED` with the
   gate or dimension that failed, not deleted.

#### Where it plugs into the loop

- **Search:** a candidate that fails a hard gate cannot become champion,
  regardless of the judge’s preference.
- **Rank:** only gate-passing episodes are ranked for training. The rank-1
  workspace is published only if it is accepted.
- **Publish:** accepted → `outputs/query{id}/workspace/`; rejected →
  `outputs/query{id}/REJECTED` with a reason and evidence.
- **Training:** rejected episodes may still be useful as negative examples, but
  they are not part of the default SFT set.

The implementation lives in `quality.rs` (`GateResult`, `DimensionScore`,
`QualityReport`, `QualityPolicy`). The pairwise judge remains for open-ended
artifacts; software tasks use the gate-plus-rubric judge.

---

## 9. Pi integration

### Provider

`~/.pi/agent/models.json`:

```json
{
  "providers": {
    "local": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        {
          "id": "qwen2.5-coder-3b-instruct-q4_k_m",
          "contextWindow": 8192,
          "maxTokens": 4096,
          "input": ["text"]
        }
      ]
    }
  }
}
```

### Executor

```bash
pi --print --mode json \
   --provider local --model <id> \
   --approve --no-context-files \
   "<task + design>"
```

`prompt_arg: true` passes the prompt as the final argument. `--mode json` yields
session events; the session JSONL is the trajectory. `--no-context-files` keeps
repository instruction files out of the run.

### Sub-agent tool

`pi/subagent.ts` registers `agent`:

```ts
pi.registerTool({
  name: "agent",
  description: "Delegate a self-contained task to a subagent. Wait for its report.",
  parameters: { role: "string", task: "string" },
  async execute(id, { role, task }, ctx) {
    // hard depth limit: 1
    // spawn `pi --print --mode json ... --system-prompt <role> <task>`
    // cwd = ctx.cwd; child session under logs/subagents/
    // parse JSONL; return final text + evidence
  }
});
```

The call is synchronous. A child failure returns an error card. There is no
background mode and no recursive fan-out.

### Session parsing

`pi_session.rs` reads JSONL entries: header, messages, assistant content
blocks, tool results, model changes. It produces the renderer's
`Message { role, content, tool_calls, loss }` shape.

### Sequential invariant

The runner executes one episode at a time. Collection is a loop, not a pool.
Subagents are foreground. No concurrency configuration exists.

---

## 10. Local serving

```bash
llama-server \
  -m ~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  --jinja -ngl 999 -c 8192 \
  --host 127.0.0.1 --port 8080
```

`--jinja` is required for Qwen tool calling. Keep 8k context. One model, one
port. Stop before training; restart after export.

---

## 11. Training stack

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft bitsandbytes datasets accelerate safetensors
```

| Setting | Pilot |
|---|---|
| Quantization | NF4 |
| Compute | FP16 |
| LoRA r / alpha | 16 / 32 |
| Window / overlap | 1024 / 128 |
| Batch / accum | 1 / 3 |
| Grad checkpointing | on |
| Optimizer | paged AdamW 8-bit |
| Steps | 100 |

Do not use BF16. Do not train while the inference server is running.

---

## 12. Testing

**Unit:** config validation, marker parsing, champion arithmetic, CPython RNG,
Bradley–Terry fixtures, 3/1 split, checkpoint selection, prompt normalization,
render masks.

**Contract:** fake runner, fake chat client, fake tokenizer; the full inner loop
runs without a model.

**Golden:** prompt bytes per version, pair schedule, BT scores, token ids and
labels.

**Integration:** one task, three designs, four episodes, ten train steps,
export, reload.

**Acceptance:**

- `cargo test --workspace` passes.
- `evolve validate` and `evolve plan` pass on the pilot config.
- `evolve cycle` completes L0 → L1.
- The exported GGUF loads and answers.
- One 3B episode stays below 7.5 GB VRAM.
- Every selected episode yields windows; no silent drops.

---

## 13. Milestones

| Milestone | Deliverable | Exit |
|---|---|---|
| M0 | Scaffold: config, design, chat, judge, CLI | tests pass, `plan` runs |
| M1 | history, champion, ranking, select, checkpoint | **done**: 53 tests |
| M2 | Pi executor + session parser | **done**: process runner, Pi session flags, JSONL parser, subagent extension; 20 tests |
| M3 | prompt + search | **done** |
| M4 | collect + rank + select | **done** |
| M5 | render | **done** (tokenizer port; byte fallback) |
| M6 | train + export | **done** as command ports |
| M7 | cycle | **done** (fake-port integration test) |
| M8 | pilot | report, memory, timing |

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| 3B judge noise | more comparisons, fixed-reference option, 7B later |
| OOM | 3B, 1k windows, batch 1, stop server |
| Pi recursion | hard depth 1, timeout |
| Pi format change | pin 1.0.2, fixture tests |
| Bad optimizer output | marker check, stop task |
| Render mismatch | golden token/label fixtures |
| License | original prompts/tasks, no copied assets |
| Full run too slow | pilot first, scale numbers later |

---

## 15. Commands

```bash
# serve
llama-server -m ~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  --jinja -ngl 999 -c 8192 --host 127.0.0.1 --port 8080

# config check
cargo run -p evolving-orchestrator -- validate \
  --config configs/evolve/local-3b.json \
  --tasks configs/evolve/tasks

# plan
cargo run -p evolving-orchestrator -- plan \
  --config configs/evolve/local-3b.json \
  --tasks configs/evolve/tasks

# one generation (after stages land)
cargo run -p evolving-orchestrator -- cycle \
  --config configs/evolve/local-3b.json \
  --tasks configs/evolve/tasks
```

---

## 16. Current state

Live: the full pipeline and all CLI stages: `config`, `design`, `chat`,
`judge`, `exec`, `pi_session`, `history`, `champion`, `ranking`, `select`,
`prompt`, `search`, `collect`, `render`, `train`, `export`, `cycle`, the Pi
`agent` extension, pilot config, and pilot task files. 53 tests pass.

Not yet run: a live Pi/llama-server episode and a real QLoRA/GGUF export. Those
are M8 (pilot) and require the local model server and training environment.
