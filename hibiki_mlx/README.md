# hibiki-mlx

## Requirements

- Python 3.12 or later
- Apple silicon, for MLX
- The project's Conda environment, `hibiki`; run Python through
  `conda run -n hibiki <command>`

Download the pinned Hibiki MLX model into the repository's ignored `artifacts/`
directory:

```python
from hibiki_mlx import download_model

artifact_directory = download_model()
```

The download is roughly 4 GB and includes the Hibiki and Mimi safetensors,
configuration, and SentencePiece tokenizer. It is pinned to the released model
revision and is not included in this package or committed to the repository.

## Checking the weights

`start()` reads the bundle's `config.json`, then checks the Mimi codec weights
followed by the Hibiki generator weights. Only the safetensors headers are read,
so nothing is allocated:

```python
from hibiki_mlx import start

mimi, hibiki = start()
print(mimi.summary())
print(hibiki.summary())
```

## Loading the model

`load_model()` runs the same check, then builds the local MLX modules and
strict-loads both files into them: the Mimi codec first, the Hibiki generator
second. Strict loading means MLX rejects the bundle unless every parameter this
implementation declares is present with the expected shape, and nothing else is.

```python
from hibiki_mlx import load_model

model = load_model()
print(model.summary())

codes = model.mimi.encode_step(pcm_frame, model.mimi.make_encoder_cache())
```

Attention caches and streaming state belong to a session, not to the loaded
model, so the codec and the generator hand them out on request.

## Translating

```shell
conda run -n hibiki python -m hibiki_mlx french.wav english.wav   # translate a file
conda run -n hibiki python -m hibiki_mlx --check                  # report the bundle and exit
```

The file is fed to the same streaming session one 80 ms frame at a time, in
order, with no lookahead; nothing about it is offline except that the input
happens to be available up front. `--temp 0` decodes greedily, which is what the
parity fixture in the tests compares against.

A session takes PCM chunks of any length and returns one result per generation
step:

```python
from hibiki_mlx import InferenceSession, load_model

model = load_model()
session = InferenceSession(model)
session.warmup()

for result in session.push_pcm(french_pcm):
    if result.text:
        print(result.text, end="")
    if result.pcm is not None:
        play(result.pcm)
results = session.finish()
```

Each `StepResult` carries `text_frame_index` and `audio_frame_index` separately.
They are not the same position: the text belongs to text frame `t`, while the
audio that becomes complete during that step is target audio frame `t - 2`,
because seven of the eight codebooks are delayed by two frames. `text_time` and
`audio_time` convert each to Model time using the codec's own frame rate.

`finish()` pads the leftover PCM chunk and pushes six frames of silence so the
delayed codebooks of the final frames can be completed. That is silence-tail
finalization, an explicit fallback rather than learned end-of-stream behaviour.

Reading a file that is not 16-bit mono WAV at 24 kHz needs `sphn`, which is
installed as a dependency; without it, such a file is refused rather than read
at the wrong rate.

## Tests

Run Python through the project's Conda environment:

```shell
conda run -n hibiki python -m unittest discover -s tests -t .
```

Tests that need the released weights read `HIBIKI_MODEL_DIR` and skip when it is
unset; they never download anything.
