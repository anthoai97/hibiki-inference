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
```

The command prints the English text and writes the translated audio. Input is
decoded and resampled as needed.

## Tests

Run Python through the project's Conda environment:

```shell
conda run -n hibiki python -m unittest discover -s tests -t .
```

Tests that need the released weights read `HIBIKI_MODEL_DIR` and skip when it is
unset; they never download anything.
