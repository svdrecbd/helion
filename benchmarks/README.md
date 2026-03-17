## Benchmarking

Performance comparison between Helion, torch.compile, Triton, and PyTorch eager is done by leveraging [TritonBench](https://github.com/meta-pytorch/tritonbench).

Currently supported kernels for performance comparison are listed in `KERNEL_MAPPINGS` in `benchmarks/run.py`.

To run the benchmark:

`$ python benchmarks/run.py --metrics speedup,accuracy --kernel <kernel_name>`

e.g. for `vector_add` kernel:

`$ python benchmarks/run.py --metrics speedup,accuracy --kernel vector_add`

## Autotuner Control-Plane Microbenchmarks

For CPU-only parity and timing checks of internal autotuner hot paths, run:

`$ python benchmarks/autotuner_hotpaths.py --repeat 10`

This harness benchmarks the current implementations against legacy reference
implementations and fails if parity does not hold. It is useful when landing
internal autotuner or compiler-control-flow optimizations that do not change
kernel math directly.

The harness can also target another checkout:

`$ python benchmarks/autotuner_hotpaths.py --repo-root /path/to/other/helion --repeat 10`

## Branch-To-Branch Comparison

For a full comparison between `main` and the current working tree, use:

`$ python benchmarks/compare_refs.py --baseline-ref main --skip-gpu`

That command runs the CPU control-plane harness against `main` and the current
checkout and writes a Markdown report plus JSON artifacts under
`benchmarks/results/`.

On a CUDA machine with `tritonbench` already available, run the broader suite:

`$ python benchmarks/compare_refs.py --baseline-ref main`

By default, that runs all supported kernels with:

`--metrics speedup,accuracy,tflops,gbps --input-sample-mode equally-spaced-k --num-inputs 20`

The extra arguments after `--` are passed straight through to
`benchmarks/run.py`, so you can choose the kernel subset, metrics, or input
sampling mode you want for PR evidence.
