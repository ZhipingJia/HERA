# HERA

**A fully heterogeneous compute-in-memory memristor chip for flexible AI processing**

This repository accompanies the manuscript *"A fully heterogeneous compute-in-memory
memristor chip for flexible AI processing"* (Nature, under review, 2026). It provides
the **affinity-aware mapping framework** that assigns each layer of a neural network to
the most suitable HERA compute tier (Analogue CIM, Digital Compute-Near-Memory, or
General-Purpose Compute), together with reproducible demos for the Faster R-CNN
object-detection and PrivateLoRA cloud-edge LLM workloads, and a supporting VGG16 study.

The code implements the Methods pipeline: per-layer accuracy-sensitivity (KLD) and
physical-benefit (EDP) profiling, affinity-score synthesis, progressive top-k scheme
construction, and objective-based selection of the **HERA-A** (accuracy-oriented) and
**HERA-P** (performance-oriented) configurations.

## License

Released under the [MIT License](LICENSE).

## Repository contents

```
hera/
  hardware/        Analytical ACIM / DCNM / GPC latency, energy and EDP models
  affinity/        Affinity ranking, scheme construction, baselines (AccDrop, GA Search)
  workloads/       Layer layouts + report builders for Faster R-CNN, PrivateLoRA, VGG16
examples/
  fasterrcnn/      Mapping demo + GPU workload: lightweight Faster R-CNN model,
                   VOC mAP evaluation, DCNM INT8 PTQ/QAT, released detector checkpoints
  privatelora/     Mapping demo + GPU workload: PrivateLoRA Llama-2-7b modeling,
                   BoolQ/GSM8K evaluation (bundled lm-evaluation-harness fork),
                   INT8/CGRA quantization
  vgg16/           Mapping demo for the VGG16 supporting study
  data/            Small synthetic profile JSONs used by the mapping demos
tests/             Smoke tests (also verify the demo outputs)
```

The repository has two tiers:

1. **Mapping demos (CPU, NumPy only)** — reproduce the paper's HERA-A / HERA-P layer
   assignments from per-layer profiles in under a second on any desktop machine. These
   demonstrate the affinity-mapping logic only; the underlying workloads (object
   detection, LLM inference, image classification) are themselves run on GPUs.
2. **GPU workloads** (`examples/fasterrcnn`, `examples/privatelora`) — the PyTorch
   inference, quantization, and evaluation code behind the manuscript's Faster R-CNN and
   PrivateLoRA results, including released Faster R-CNN checkpoints that reproduce the
   detection accuracies in Fig. 4f.

### What is **not** included

This repository does **not** contain:

- the infrared human-detection dataset (VOC-style layout documented below; available from
  the corresponding authors upon reasonable request);
- the Llama-2-7b base model (download `meta-llama/Llama-2-7b-hf` from the Hugging Face Hub
  under the Llama 2 license);
- the PrivateLoRA adapter checkpoints in the git tree (~647 MB each) — they are published
  as [GitHub Release assets](https://github.com/ZhipingJia/HERA/releases/tag/v1.0.0); see
  `examples/privatelora/weights/README.md` for download links and SHA-256 manifests;
- the measured on-chip ACIM noise table used by the PrivateLoRA noise simulation
  (proprietary hardware characterization; the code raises a clear error and accepts a
  user-provided table), and HERA-silicon driver code.

Full on-chip functionality requires the HERA hardware platform and can be provided by the
corresponding authors upon reasonable request.

## System Requirements

**Software dependencies**

- Mapping demos (CPU): Python ≥ 3.9, NumPy ≥ 1.21 (see [`requirements.txt`](requirements.txt)).
  No GPU, compiler, or other build tooling is required.
- GPU workloads: additionally PyTorch ≥ 2.1 with CUDA, torchvision, transformers, datasets,
  evaluate and the other packages in [`requirements-gpu.txt`](requirements-gpu.txt).

**Tested on**

- Mapping demos: Python 3.11.11 / NumPy 1.26.0, Linux (kernel 5.15). Platform-independent;
  expected to run unchanged on macOS and Windows.
- GPU workloads: Python 3.11.14, PyTorch 2.5.1 + CUDA 12.1, torchvision 0.20.1,
  transformers 4.57.3, datasets 4.4.2, on Linux with NVIDIA H100 80GB GPUs. Any CUDA GPU
  with ≥ 4 GB memory suffices for Faster R-CNN; PrivateLoRA loads Llama-2-7b in FP32 and
  needs a ≥ 40 GB GPU (or adapt to half precision).

**Non-standard hardware**

- None for the mapping framework and demos — they run on a normal desktop CPU. The GPU
  workloads need a standard CUDA GPU. The HERA memristor chip (the non-standard hardware)
  is required only for the end-to-end on-chip energy/latency measurements reported in the
  manuscript; it is not needed to reproduce the mapping decisions or the detection /
  language-task accuracies here.

## Installation Guide

```bash
git clone https://github.com/ZhipingJia/HERA.git
cd HERA
python -m pip install -r requirements.txt       # mapping demos (NumPy only)
python -m pip install -r requirements-gpu.txt   # optional: GPU workloads
```

No build or compilation step is needed; the `hera` package is imported directly from the
source tree (the example scripts add the repository root to `sys.path` automatically).

**Typical install time on a normal desktop computer:** under 1 minute for the mapping demos
(NumPy only). The optional GPU stack downloads PyTorch/CUDA wheels (~3 GB) and typically
takes 5–15 minutes depending on network speed.

## Demo

The demos read a small synthetic per-layer profile (illustrative KLD / EDP placeholders, not
measured data — see [`examples/data/README.md`](examples/data/README.md)) and print the
HERA-A and HERA-P tier assignments produced by the affinity framework.

**Faster R-CNN**

```bash
python examples/fasterrcnn/reproduce_mapping.py \
  --profiles-json examples/data/fasterrcnn_synthetic_profile.json
```

Expected output:

```
HERA-A (7 ACIM / 7 DCNM):
  ACIM: extractor.0.2, extractor.1.1, extractor.1.3, extractor.2.0, extractor.2.2, extractor.2.4, rpn.conv1
  DCNM: extractor.0.0, rpn.score, rpn.loc, head.classifier.0, head.classifier.2, head.cls_loc, head.score
HERA-P (9 ACIM / 5 DCNM):
  ACIM: extractor.0.2, extractor.1.1, extractor.1.3, extractor.2.0, extractor.2.2, extractor.2.4, rpn.conv1, head.classifier.0, head.classifier.2
  DCNM: extractor.0.0, rpn.score, rpn.loc, head.cls_loc, head.score
```

This matches Extended Data Fig. 8: the deeper extractor convolutions and the shared
`rpn.conv1` are mapped to ACIM in both objectives, the first convolution and the RPN/ROI
output branches stay on DCNM, and the two ROI-head classifier layers (`head.classifier.0`,
`head.classifier.2`) are the swing layers — kept on DCNM by HERA-A and moved to ACIM by HERA-P.

**PrivateLoRA**

```bash
python examples/privatelora/reproduce_mapping.py \
  --task boolq --profiles-json examples/data/privatelora_boolq_synthetic_profile.json
python examples/privatelora/reproduce_mapping.py \
  --task gsm8k --profiles-json examples/data/privatelora_gsm8k_synthetic_profile.json
```

Expected output (per the layer counts in Fig. 5c; `--list-dcnm` additionally prints the
retained layer names):

```
Task: boolq
HERA-A (80 ACIM / 17 DCNM):
HERA-P (88 ACIM / 9 DCNM):
```
```
Task: gsm8k
HERA-A (76 ACIM / 21 DCNM):
HERA-P (84 ACIM / 13 DCNM):
```

The DCNM counts include the Head LB layer, which is always retained on DCNM. Thus BoolQ
HERA-A maps 80 of the 96 PLM matrices to ACIM (16 PLM + Head LB on DCNM) and HERA-P maps 88
(8 PLM + Head LB on DCNM); GSM8K retains 20 and 12 PLM matrices on DCNM respectively.

**VGG16**

```bash
python examples/vgg16/reproduce_mapping.py \
  --profiles-json examples/data/vgg16_synthetic_profile.json
```

Only the first convolution (`features.0`) has a negative ACIM EDP benefit and is kept on
DCNM; the remaining layers are mapped to ACIM.

**Expected run time for each demo on a normal desktop computer:** under 1 second (≈ 0.2 s
measured on the test machine).

## Instructions for Use

### Run on your own data

Each demo consumes a JSON list of per-layer profile rows with the following schema:

| field | meaning |
|---|---|
| `name` | full layer name (must match the workload's layer layout) |
| `short_name` | compact display name |
| `kld` | per-layer accuracy sensitivity (KL divergence under single-layer ACIM substitution) |
| `edp_acim` | per-layer energy-delay product when mapped to ACIM |
| `edp_dcnm` | per-layer energy-delay product when mapped to DCNM |

Supply your own profile JSON via `--profiles-json` to map a different network. The two HERA
objectives are controlled by `--hera-a-dcnm-retain` / `--hera-p-dcnm-retain`, i.e. the number
of lowest-affinity layers kept on DCNM (Methods, Step 3, bottom-K labelling); `0` maps every
affinity-qualified layer to ACIM.

To generate the `kld` and `edp_*` columns from your own model:

- **EDP** — call `hera.affinity.profiling.profile_edp` (or the per-workload helpers in
  `hera/workloads/*/profiling.py`) with your layer shapes and hardware parameters
  (`ACIMParameters`, `DCNMParameters`). The analytical model is in
  `hera/hardware/analytical_models.py`.
- **KLD** — run your own pure-DCNM vs single-layer-ACIM logits and pass them to
  `hera.affinity.profiling.compute_kld_from_logits`, then merge with the EDP rows via
  `merge_kld_edp_profiles`.

### Programmatic use

```python
from hera.affinity import compute_affinity_records, build_scheme_family, select_objective_scheme
from hera.affinity.core import LayerProfile

profiles = [LayerProfile("L0", "L0", kld=0.01, edp_acim=1.0, edp_dcnm=2.0), ...]
rank = compute_affinity_records(profiles, require_positive_edp_diff=True)
schemes = build_scheme_family(rank)
hera_a = select_objective_scheme(schemes, "HERA-A", dcnm_retain=1)
hera_p = select_objective_scheme(schemes, "HERA-P", dcnm_retain=0)
```

The `AccDrop` and `GA Search` baselines from the Methods are available in
`hera.affinity.baselines`.

## GPU Workloads

### Faster R-CNN (infrared human detection)

`examples/fasterrcnn/` contains the paper's lightweight channel-32 detector
(`FasterRCNNVGG16LIGHTV3`, 14 Conv2d/Linear target layers) and the LSQ INT8 quantization
layers, with inference, VOC mAP evaluation, DCNM INT8 PTQ and QAT. Released checkpoints
(each ~320 KB) live in `examples/fasterrcnn/weights/` — see the README there for the full
list; each file is a sanitized `{"model": state_dict, "meta": {...}}` payload.

The infrared dataset is not distributed. To evaluate, point `--voc-data-dir` at a
VOC-style root containing `imgs/<id>.jpg`, `Anotations/All_In_One_Anot_voc/<id>.xml`
(VOC bounding-box XML), `list_files/test.list` (image ids), and the
`slice_info_w320_h240.json` crop index used by the `valid_crop` test transform.

```bash
cd examples/fasterrcnn

# mAP evaluation — all-DCNM INT8 reference (deterministic)
python eval_voc.py --checkpoint weights/fasterrcnn_all_dcnm_int8_qat.pth     --quant-config configs/config_dcnm_int8.yaml --voc-data-dir /path/to/dataset

# mAP evaluation — hybrid HERA-A / HERA-P / all-ACIM (ACIM layers inject sampled noise)
python eval_voc.py --checkpoint weights/fasterrcnn_hera_a_scheme7.pth     --effective-config configs/effective_config_hera_a_scheme7.json --voc-data-dir /path/to/dataset

# DCNM INT8 PTQ from the full-precision checkpoint
python build_dcnm_baseline.py --fp-checkpoint weights/fasterrcnn_fp.pth     --output-dir runs/dcnm_ptq --voc-data-dir /path/to/dataset

# DCNM / hybrid-ACIM quantization-aware training
python train_dcnm_int8_qat.py --source-checkpoint runs/dcnm_ptq/fasterrcnn_dcnm_int8_from_fp_best.pth     --output-root runs/qat --voc-data-dir /path/to/dataset
```

Detection accuracies reproduced with this code on the paper's evaluation split
(1,276 images; expected run time ≈ 10 s per evaluation on one modern GPU):

| Checkpoint | mAP50 (VOC07) |
|---|---|
| all-DCNM INT8 QAT | 0.8892 |
| HERA-A (scheme7) | 0.8859 |
| HERA-P (scheme9) | 0.8512 |
| all-ACIM | 0.5428 |

The all-DCNM evaluation is deterministic; the hybrid configurations sample ACIM noise
at evaluation time, so their mAP varies slightly across seeds.

### PrivateLoRA (cloud-edge collaborative LLM)

`examples/privatelora/` contains the PrivateLoRA Llama-2-7b model definition (96 PLM
`lora_mobile` matrices + Head LB), the INT8/CGRA quantization adapter, and a bundled
lm-evaluation-harness fork for BoolQ / GSM8K evaluation.

Setup: download `meta-llama/Llama-2-7b-hf`, download the PrivateLoRA checkpoints from the
[v1.0.0 release](https://github.com/ZhipingJia/HERA/releases/tag/v1.0.0) (SHA-256 manifests
in `examples/privatelora/weights/README.md`), then:

```bash
cd examples/privatelora
export PRIVATE_LLM_BASE_MODEL=/path/to/llama-2-7b-hf   # or the HF Hub id
# optional: PRIVATE_LLM_DATASET_ROOT=/path/to/datasets_save_to_disk_cache
# (BoolQ/GSM8K are otherwise fetched from the Hugging Face Hub)

# BoolQ accuracy (forward-only; ~4 min on one H100)
python eval_pl.py --checkpoint-path /path/to/pl_boolq_fp_r256.bin --lora-rank 256     --tasks boolq --batch-size 16 --output-json results/boolq_fp.json

# GSM8K exact-match (greedy generation; ~1.5 h on one H100)
python eval_pl.py --checkpoint-path /path/to/pl_gsm8k_fp_r256.bin --lora-rank 256     --tasks gsm8k_yaml --batch-size 16 --output-json results/gsm8k_fp.json
```

Accuracies reproduced with this code (full evaluation sets):

| Task | Reproduced | Recorded best during training |
|---|---|---|
| BoolQ (3,270 samples) | 0.8862 | 0.8865 |
| GSM8K (1,319 problems) | 0.2729 | 0.2691 |

## Tests

```bash
python tests/test_smoke.py     # or: pytest
```

The smoke tests exercise the analytical hardware model and the KLD divergence, and assert
that the demos reproduce the manuscript's HERA-A / HERA-P splits. They run in under 1 second.

## Citation

Citation information will be added here upon publication.

## Contact

For questions regarding the code or the manuscript, please contact the corresponding authors
listed in the paper.
