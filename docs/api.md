# API

This document outlines the APIs for running experiments.

## Performance experiment API

We provide a context manager-based API for performance experiments. Below is a minimal example demonstrating how to use this API.

```python
from pimba import Performance

with Performance.setup(
    model="mamba2-2.7b",
    system="Pimba",
    num_gpus=1,
    batch_size=32,
    gpu="A100",
) as exp:
    exp.print_config()
    print(exp.run())
```

## Accuracy experiment API

The same applies to the accuracy experiment API.. Below is a minimal example demonstrating how to use this API.

```python
from pimba import Performance

with Accuracy.setup(
    model="mamba2-2.7b",
    quant="mx8",
    use_sr=True,
    datasets=["wikitext"],
) as exp:
    exp.print_config()
    print(exp.run())
```

