# Challenges in reproducibility

While preparing the artifact evaluation code, we encountered several reproducibility challenges. Through engineering efforts, we identified two key issues.

## Triton's autotune issue

The first issue was related to Triton’s autotuning mechanism. Many recent works leverage Triton to implement custom GPU kernels, and we adopted model code from repositories such as Hugging Face Transformers and Flash Linear Attention, which also rely on Triton.

During our accuracy evaluations, we observed small but consistent numerical discrepancies across repeated runs. This was caused by Triton’s autotuning feature, which compiles several kernel variants with different hyperparameters (e.g., num_warps) and benchmarks them at runtime to select the fastest one. Although all variants are theoretically functionally equivalent, slight differences in floating-point addition orders introduced numerical variations. These variations, while negligible in general, had a noticeable impact in our quantization-sensitive environment.

We resolved the issue by disabling autotuning to ensure deterministic kernel selection, which allowed us to reproduce accuracy results reliably on the same hardware. However, since Triton does not provide a built-in method to disable autotuning, we monkey-patched it by overriding the autotune logic with a dummy function, as shown below:

```python
import triton


def _dummy_tuner(**kwargs):
    def decorator(fn):
        return fn

    return decorator


triton.autotune = _dummy_tuner
```

## Python's multiprocessing issue

The second issue stemmed from Python’s multiprocessing module. To speed up performance measurements, we parallelized experiments using a multiprocessing.Pool. However, we found that some performance results were not reproducible. After debugging, we identified the root cause: the pool reuses worker processes across tasks. Since we employed caching to improve performance, intermediate values cached during one experiment could inadvertently be reused in subsequent runs assigned to the same worker process.

We addressed this issue by configuring the pool such that each process handled only a single task and was terminated afterward. This eliminated unintended cache reuse and restored reproducibility. The following code snippet shows how we implemented this:

```python
from multiprocessing import Pool

with Pool(maxtasksperchild=1) as pool:
    perf_res = [
        result
        for result in tqdm(
            pool.imap_unordered(perf_exp, exp_list),
            desc="performance experiments",
            total=len(exp_list),
        )
    ]

```
