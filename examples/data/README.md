# Demo Profiles

This directory contains small per-layer profile JSON files that drive the CPU
mapping demos (`examples/*/reproduce_mapping.py`). Each file covers the full
layer set of its workload (Faster R-CNN: 14 layers; PrivateLoRA: 96 PLM matrices
+ Head LB; VGG16: 16 layers).

## What is real and what is illustrative

* **The hardware model is real.** The per-layer EDP is defined by the calibrated
  analytical ACIM / DCNM model in `hera/hardware/` (coefficients in
  `hera/hardware/hera_config.py`). The GPU workloads under `examples/fasterrcnn`
  and `examples/privatelora` use exactly this model — via the same shared
  coefficients — to produce the per-layer EDP behind the manuscript's results.
* **These demo profiles are illustrative inputs.** The `kld` and `edp_*` values
  in the JSON files here are representative numbers chosen so the mapping
  framework reproduces, in under a second on a CPU, the qualitative HERA-A /
  HERA-P split each workload reports (e.g. Faster R-CNN keeps the ROI-head
  classifier layers on DCNM under HERA-A and moves them to ACIM under HERA-P).
  They are not paper-reported measurements and are not derived from private
  datasets, checkpoints, or training logs.

To produce real per-layer EDP from a network, build `LayerShape` objects and call
`hera.hardware.compare_acim_dcnm` (or run the GPU workloads' profiling scripts).

## Schema

Each profile row uses the minimal schema consumed by the demo scripts:

- `name`: full layer name
- `short_name`: compact display name
- `kld`: per-layer accuracy sensitivity (illustrative)
- `edp_acim`: per-layer ACIM energy-delay product (illustrative)
- `edp_dcnm`: per-layer DCNM energy-delay product (illustrative)
