# Local 35B MoE

Qwen3.5-35B-A3B is a 35B-total, ~3B-active MoE. The Q4_K_M GGUF is **22.29 GB**
of weights. Your 62 GB of RAM holds it; the 8 GB GPU holds attention,
embeddings, output, and KV cache while the experts stay on CPU.

This model is for **inference and task mode**. It cannot be QLoRA-trained on
8 GB: the 4-bit weights alone are larger than VRAM, before activations and
optimizer state.

## Serve it

```bash
llama-server \
  -m ~/models/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --alias Qwen3.5-35B-A3B \
  --jinja \
  -ngl 999 \
  -cmoe \
  -c 65536 \
  -t 12 \
  --host 127.0.0.1 \
  --port 8080
```

`-cmoe` keeps all MoE expert weights in system RAM. `-ngl 999` offloads the
non-MoE tensors to the GPU. On a llama.cpp build without `-cmoe`, use
`--n-cpu-moe 40`.

If VRAM is tight, reduce `-c` to `16384` or `-ngl` to `30`. If VRAM has
headroom, move some experts back to the GPU with `-ncmoe 30` (first 30 layers
on CPU, last 10 on GPU).

Verify:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq
```

## Point Pi at it

Copy `configs/evolve/pi-models.json` to `~/.pi/agent/models.json` (or merge the
provider). The model ID must be exactly `Qwen3.5-35B-A3B`, matching `--alias`
and `model_id`.

## Run task mode

```bash
evolve cycle \
  --config configs/evolve/local-35b.json \
  --tasks configs/evolve/tasks \
  --task-only
```

`--task-only` runs search, collection, ranking, selection, and publishing, then
stops before rendering and training. The best workspace is written to:

```text
runs/L0-35b/outputs/query1/workspace
```

Do not run the full `cycle` without a training command and a model that fits
the GPU. The 35B model is the planner/executor/judge here; a separate smaller
student would be required for local weight updates.
