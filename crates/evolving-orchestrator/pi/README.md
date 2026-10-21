# Pi sub-agent extension

Pi intentionally ships without sub-agents. `subagent.ts` adds one synchronous
`agent` tool so a design block can delegate work.

## Load it

```bash
pi -e crates/evolving-orchestrator/pi/subagent.ts \
  --provider local --model qwen2.5-coder-3b-instruct-q4_k_m \
  --print --mode json \
  "Your task and design block"
```

The extension is loaded into the orchestrator. Child processes are started with
`--no-extensions`, so they cannot delegate again. Depth is hard-capped at one.

## Environment

| Variable | Meaning |
|---|---|
| `EVOLVE_MODEL` | Required. Model id understood by Pi. |
| `EVOLVE_PROVIDER` | Provider name. Default `local`. |
| `EVOLVE_LOGS` | Episode logs directory. Child sessions go to `<EVOLVE_LOGS>/subagents`. |
| `EVOLVE_PI_BIN` | Pi executable. Default `pi`. |
| `EVOLVE_SUBAGENT_DEPTH` | Internal. Starts at 0. |
| `EVOLVE_SUBAGENT_TIMEOUT_MS` | Child timeout. Default 900000 (15 min). |

## Behavior

- The child runs in the orchestrator's current working directory.
- The role is passed as `--system-prompt`.
- The task is passed as the prompt.
- The child writes a session JSONL under the session directory; `evolve` parses
  it with `pi_session.rs`.
- The call is synchronous. There is no background mode.
- A timeout, nonzero exit, or `error`/`aborted` stop reason returns an error
  tool result with the child's stderr tail.
