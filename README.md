# Pimba: A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving

This repository contains the performance simulator and accuracy evaluation code developed for the Pimba project, accepted at MICRO'25.

## Hardware requirements

- `GPU` for accuracy evaluation
    - An NVIDIA GPU based on the Ampere architecture or newer, equipped with at least 24GB of memory
    - For example, RTX 3090, RTX 4090, RTX 5090, A6000, A100, etc.
- `Storage` for accuracy evaluation
    - At least 100GB of available storage space to accommodate model weights and datasets

## Building the projects

We use `uv` to manage dependencies and build tools. With the following commands, `uv` automatically downloads the required dependencies and build tools (e.g., cmake, ninja) in exactly the same versions we used and compiles our project.

```sh
uv sync
uv run cmake --preset release
uv run cmake --build build
```

## Reproducing the experiment results

You can run all experiments easily using the following command:

```sh
uv run python scripts/run.py
```

This command generates two files, `accuracy_result.yaml` and `performance_result.yaml`, under the `res/` directory, which are then used to reproduce the figures and the table. Using the result files, you can simply reproduce the figures and the table in the paper by executing the following command:

```sh
uv run python scripts/draw.py
```

This process generates PDF files for the figures and a CSV file for the table in the `summary/` directory.

## Further readings

We also document the challenges we encountered while preparing the code and a breif API reference to assist those who wish to extend our codebase.

- `docs/reproducibility.md` documents the challenges we faced while preparing the code.
- `docs/api.md` provides a brief API reference to assist those who wish to extend our codebase.

## Credits

Our code is adapted and extended from the following sources:
- GLA, HGRN2, and RetNet model code from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- LLaMA, OPT, and Zamba2 model code from [transformers](https://github.com/huggingface/transformers)
- Mamba2 model code from [mamba](https://github.com/state-spaces/mamba)
- memory simulation code from [ramulator2](https://github.com/CMU-SAFARI/ramulator2)
- system performance simulation code from [attacc_simulator](https://github.com/scale-snu/attacc_simulator)

## Citation

If you want to cite our work, please cite our paper using the following BibTeX entry:

```bib
@misc{pimba,
  title         = {{Pimba: A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving}},
  author        = {Wonung Kim and Yubin Lee and Yoonsung Kim and Jinwoo Hwang and Seongryong Oh and Jiyong Jung and Aziz Huseynov and Woong Gyu Park and Chang Hyun Park and Divya Mahajan and Jongse Park},
  year          = {2025},
  url           = {https://arxiv.org/abs/2507.10178},
  eprint        = {2507.10178},
  archiveprefix = {arXiv},
  primaryclass  = {cs.AR}
}
```
