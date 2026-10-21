# Contributing

Evolving Orchestrator is a Rust workspace. Read [`AGENTS.md`](AGENTS.md) for the rules and
[`crates/evolving-orchestrator/BLUEPRINT.md`](crates/evolving-orchestrator/BLUEPRINT.md) for the end-to-end
design.

## Requirements

- Rust and Cargo (edition 2021).
- `git`.
- An OpenAI-compatible model server.
- `pi` for the executor, or any command that accepts a prompt.
- Python, PyTorch, PEFT, and bitsandbytes for training.

## Build and test

```bash
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

## Check the configuration

```bash
cargo run -p evolving-orchestrator -- validate \
  --config configs/evolve/local-3b.json \
  --tasks configs/evolve/tasks

cargo run -p evolving-orchestrator -- plan \
  --config configs/evolve/local-3b.json \
  --tasks configs/evolve/tasks
```

`validate` checks the cycle configuration, the task split, the endpoint, and
the task files. `plan` prints the model, seeds, counts, training
hyperparameters, and stage order without side effects.

## Adding a stage

1. Define the pure decision as a function with no I/O.
2. Put external effects behind a port in `engine`-style leaf code.
3. Add a fake port implementation for tests.
4. Add a golden fixture if the stage produces deterministic text or numbers.
5. Wire the stage into `cycle.rs` and the CLI.
6. Update `BLUEPRINT.md` in the same change.

## Quality bar

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`

A commit that describes unshipped behavior in a document is a bug. Update the
document or drop the claim.

## Sequential pipeline

Do not add concurrency flags, worker pools, or background agents. The pipeline
runs one episode at a time, one inference server, one model. Collection is a
loop.

## License

MIT. By contributing, you agree that your changes are licensed under the same
terms.
