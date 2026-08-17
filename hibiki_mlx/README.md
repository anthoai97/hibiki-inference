# hibiki-mlx

## Requirements

- Python 3.12 or later
- Apple silicon, for MLX
- The project's Conda environment, `hibiki`; run Python through
  `conda run -n hibiki <command>`

Download the model:

```python
from hibiki_mlx import download_model

download_model()
```

## Translating

```shell
conda run -n hibiki python -m hibiki_mlx french.wav english.wav   # translate a file
conda run -n hibiki python -m hibiki_mlx --play french.wav        # play as it translates
conda run -n hibiki python -m hibiki_mlx --metrics french.wav     # monitor each step
```

The command prints English text as it is decoded and writes the translated
audio. Pass `--play` to stream decoded English audio to the default sound
device. Input is decoded and resampled as needed.

`--metrics` reports codec, generation, target-decoding, and text-decoding time
for each generation step, along with MLX allocator memory and process peak RSS.

## Quantized artifacts

Create a separate Q8 or Q4 bundle; the BF16 source bundle is never modified:

```shell
conda run -n hibiki python -m hibiki_mlx.quantize artifacts/hibiki-1b-mlx-bf16 artifacts/hibiki-1b-mlx-q8 --bits 8
conda run -n hibiki python -m hibiki_mlx.quantize artifacts/hibiki-1b-mlx-bf16 artifacts/hibiki-1b-mlx-q4 --bits 4
```

Validate the output and then publish it with Hugging Face authentication from
`huggingface-cli login` or `HF_TOKEN`:

```shell
conda run -n hibiki python -m hibiki_mlx.upload artifacts/hibiki-1b-mlx-q8 --repo-id YOUR_ACCOUNT/hibiki-1b-mlx-q8 --private --dry-run
conda run -n hibiki python -m hibiki_mlx.upload artifacts/hibiki-1b-mlx-q8 --repo-id YOUR_ACCOUNT/hibiki-1b-mlx-q8 --private
```

## BF16 versus Q8 benchmark

Benchmark every WAV below `assets/` with clean BF16 and Q8 worker processes.
Each model is loaded and warmed once, then each input uses a fresh inference
session. The command writes per-file and aggregate timing, real-time-factor,
transcript, and MLX-memory measurements to JSON and CSV; it does not write
generated audio.

```shell
conda run -n hibiki python -m hibiki_mlx.benchmark \
  --bf16-artifacts artifacts/hibiki-1b-mlx-bf16 \
  --q8-artifacts artifacts/hibiki-1b-mlx-q8 \
  --assets assets \
  --output-dir benchmarks/bf16-vs-q8
```

Omit `--output-dir` to create a timestamped directory under `benchmarks/`.

## Tests

Run Python through the project's Conda environment:

```shell
conda run -n hibiki python -m unittest discover -s tests -t .
```

Tests that need the released weights read `HIBIKI_MODEL_DIR` and skip when it is
unset; they never download anything.
