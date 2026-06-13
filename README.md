# HERA

**A fully heterogeneous compute-in-memory memristor chip for flexible AI processing**

This repository accompanies the manuscript *"A fully heterogeneous compute-in-memory
memristor chip for flexible AI processing"* (Nature, under review, 2026). Its core is the
**affinity-aware mapping framework** that assigns each layer of a neural network to the most
suitable HERA compute tier — Analogue CIM (ACIM), Digital Compute-Near-Memory (DCNM), or
General-Purpose Compute (GPC) — by jointly weighing per-layer accuracy sensitivity and
hardware benefit.

The framework is validated end-to-end on **VGG16 / CIFAR-100**, the audit network of the
Methods *Affinity-aware mapping* section (Fig. 3). Two further workloads — **Faster R-CNN**
object detection (Fig. 4) and **PrivateLoRA** cloud-edge LLM inference (Fig. 5) — are
included as on-chip application demonstrations; they run primarily as inference tasks and
draw on parts of the same mapping methodology.

The framework implements the Methods pipeline: per-layer accuracy-sensitivity (KLD) and
physical-benefit (EDP) profiling, affinity-score synthesis
`Affinity(l) = ΔEDP^l / (KLD^l + ε)`, progressive top-k scheme construction, and
objective-based selection of the **HERA-A** (accuracy-oriented) and **HERA-P**
(performance-oriented) configurations, with **AccDrop** and **GA Search** baselines.

## License

Released under the [MIT License](LICENSE).

## Repository contents

```
hera/                  Core affinity-aware mapping framework (the manuscript's methodology)
  hardware/            Analytical ACIM / DCNM / GPC latency, energy and EDP models
  affinity/            Affinity ranking, scheme construction, baselines (AccDrop, GA Search)
examples/
  vgg16/               ★ Framework audit network: VGG16 + CIFAR-100, real KLD / EDP
                         profiling, affinity ranking, three-method scheme families (Fig. 3)
  fasterrcnn/          Application: lightweight Faster R-CNN detection, VOC mAP evaluation,
                         DCNM INT8 PTQ/QAT, released detector checkpoints
  privatelora/         Application: PrivateLoRA Llama-2-7b, BoolQ/GSM8K evaluation
                         (bundled lm-evaluation-harness fork), INT8/CGRA quantization
  data/                Synthetic profile for the Faster R-CNN mapping demo
tests/                 Smoke tests (also verify the demo outputs)
```

### What is **not** included

This repository does **not** contain:

- the infrared human-detection dataset for Faster R-CNN (VOC-style layout documented below;
  available from the corresponding authors upon reasonable request);
- the Llama-2-7b base model (download `meta-llama/Llama-2-7b-hf` from the Hugging Face Hub
  under the Llama 2 license);
- large checkpoints in the git tree — the PrivateLoRA adapters (~647 MB each) and the VGG16
  INT8 baseline (~59 MB) are published as
  [GitHub Release assets](https://github.com/ZhipingJia/HERA/releases); see the per-example
  `weights/README.md` for download links and SHA-256 manifests;
- the measured on-chip ACIM noise table used by the noise simulation (proprietary hardware
  characterization; the code raises a clear error and accepts a user-provided table), and
  HERA-silicon driver code.

Full on-chip functionality requires the HERA hardware platform and can be provided by the
corresponding authors upon reasonable request.

## System Requirements

**Software dependencies**

- Mapping demo (CPU): Python ≥ 3.9, NumPy ≥ 1.21 (see [`requirements.txt`](requirements.txt)).
  No GPU, compiler, or other build tooling is required.
- GPU profiling / workloads: additionally PyTorch ≥ 2.1 with CUDA, torchvision, transformers,
  datasets, evaluate and the other packages in [`requirements-gpu.txt`](requirements-gpu.txt).

**Tested on**

- CPU demo: Python 3.11.11 / NumPy 1.26.0, Linux (kernel 5.15). Platform-independent;
  expected to run unchanged on macOS and Windows.
- GPU: Python 3.11.14, PyTorch 2.5.1 + CUDA 12.1, torchvision 0.20.1, transformers 4.57.3,
  datasets 4.4.2, on Linux with NVIDIA H100 80GB GPUs. VGG16 and Faster R-CNN run on any
  CUDA GPU with ≥ 4 GB memory; PrivateLoRA loads Llama-2-7b in FP32 and needs a ≥ 40 GB GPU.

**Non-standard hardware**

- None for the framework, demos, VGG16, or Faster R-CNN — they run on a normal desktop CPU
  (CPU demo) or a standard CUDA GPU. The HERA memristor chip (the non-standard hardware) is
  required only for the end-to-end on-chip energy/latency measurements reported in the
  manuscript; it is not needed to reproduce the mapping decisions or task accuracies here.

## Installation Guide

```bash
git clone https://github.com/ZhipingJia/HERA.git
cd HERA
python -m pip install -r requirements.txt       # CPU mapping demo (NumPy only)
python -m pip install -r requirements-gpu.txt   # optional: GPU profiling / workloads
```

No build or compilation step is needed; the `hera` package is imported directly from the
source tree (the example scripts add the repository root to `sys.path` automatically).

**Typical install time on a normal desktop computer:** under 1 minute for the CPU demo
(NumPy only). The optional GPU stack downloads PyTorch/CUDA wheels (~3 GB) and typically
takes 5–15 minutes depending on network speed.

## Affinity-Aware Mapping Framework — VGG16 / CIFAR-100

`examples/vgg16/` is the framework's audit network and reproduces Fig. 3. It ships the real
per-layer profiles (under `examples/vgg16/data/`) produced by the GPU pipeline below.

### Demo: reproduce the mapping on a normal desktop (CPU, seconds)

```bash
python examples/vgg16/reproduce_mapping.py
```

Reads the real VGG16-CIFAR100 KLD + EDP profiles, synthesizes the per-layer affinity
ranking (Fig. 3c), and prints the HERA-A / HERA-P assignments. Expected output:

```
VGG16 / CIFAR-100 - 16 layers, 15 ACIM-qualified (positive delta-EDP)
...
  excluded (delta-EDP <= 0, kept on DCNM): features.0
HERA-A (14 ACIM / 2 DCNM): ... DCNM: features.0, classifier.6
HERA-P (15 ACIM / 1 DCNM): ... DCNM: features.0
```

The first convolution `features.0` has limited row parallelism, so its ACIM EDP benefit is
negative and it is kept on DCNM — exactly the spatial-variation point made in Fig. 3c.
**Expected run time:** under 1 second.

### Full GPU pipeline (regenerate the profiles)

CIFAR-100 is fetched by torchvision; the full-precision VGG16-BN weights are the public
[chenyaofo](https://github.com/chenyaofo/pytorch-cifar-models) checkpoint (auto-downloaded);
the INT8 baseline (the KLD-profiling reference) is a released asset
(`examples/vgg16/weights/README.md`).

```bash
cd examples/vgg16
python eval.py --mode fp        # FP baseline accuracy (CIFAR-100)
python eval.py --mode int8      # DCNM INT8 baseline accuracy
python profile_edp.py           # per-layer ACIM/DCNM EDP  -> data/vgg16_edp_profile_cifar100.json
python profile_kld.py           # single-layer ACIM KLD    -> data/vgg16_kld_cifar100.json
python build_affinity.py        # affinity ranking         -> data/vgg16_affinity_rank_cifar100.json
python build_schemes.py         # affinity/AccDrop/GA scheme families + hardware EDP (Fig. 3d)
```

Reproduced VGG16 / CIFAR-100 accuracies with this code:

| Configuration | top-1 accuracy |
|---|---|
| FP baseline (chenyaofo) | 71.34% |
| DCNM INT8 baseline | 73.34% |

Per-scheme trained accuracy (the precision axis of Fig. 3d) is read from the published
results JSON; scheme training is not part of this release.

### Run on your own network

`reproduce_mapping.py` and the Faster R-CNN demo consume a JSON list of per-layer rows:

| field | meaning |
|---|---|
| `name` / `short_name` | layer identifier |
| `kld` | per-layer accuracy sensitivity (KL divergence under single-layer ACIM substitution) |
| `edp_acim` / `edp_dcnm` | per-layer energy-delay product on ACIM / DCNM |

Or call the framework directly:

```python
from hera.affinity.core import (
    LayerProfile, compute_affinity_records, build_scheme_family, select_objective_scheme,
)

profiles = [LayerProfile("L0", "L0", kld=0.01, edp_acim=1.0, edp_dcnm=2.0), ...]
rank = compute_affinity_records(profiles, require_positive_edp_diff=True)
schemes = build_scheme_family(rank)
hera_a = select_objective_scheme(schemes, "HERA-A", dcnm_retain=1)
hera_p = select_objective_scheme(schemes, "HERA-P", dcnm_retain=0)
```

EDP comes from the calibrated analytical model in `hera.hardware`; the `AccDrop` and
`GA Search` baselines are in `hera.affinity.baselines`.

## Applications

### Faster R-CNN (infrared human detection)

`examples/fasterrcnn/` contains the paper's lightweight channel-32 detector
(`FasterRCNNVGG16LIGHTV3`, 14 Conv2d/Linear target layers) with inference, VOC mAP
evaluation, DCNM INT8 PTQ and QAT. Released checkpoints (~320 KB each) are in
`examples/fasterrcnn/weights/`. The infrared dataset is not distributed; point
`--voc-data-dir` at a VOC-style root (`imgs/<id>.jpg`,
`Anotations/All_In_One_Anot_voc/<id>.xml`, `list_files/test.list`,
`slice_info_w320_h240.json`).

```bash
cd examples/fasterrcnn
# all-DCNM INT8 reference (deterministic)
python eval_voc.py --checkpoint weights/fasterrcnn_all_dcnm_int8_qat.pth     --quant-config configs/config_dcnm_int8.yaml --voc-data-dir /path/to/dataset
# hybrid HERA-A / HERA-P / all-ACIM (ACIM layers inject sampled noise)
python eval_voc.py --checkpoint weights/fasterrcnn_hera_a_scheme7.pth     --effective-config configs/effective_config_hera_a_scheme7.json --voc-data-dir /path/to/dataset
# DCNM INT8 PTQ then QAT
python build_dcnm_baseline.py --fp-checkpoint weights/fasterrcnn_fp.pth     --output-dir runs/dcnm_ptq --voc-data-dir /path/to/dataset
python train_dcnm_int8_qat.py --source-checkpoint runs/dcnm_ptq/fasterrcnn_dcnm_int8_from_fp_best.pth     --output-root runs/qat --voc-data-dir /path/to/dataset
```

Detection accuracies reproduced on the paper's evaluation split (1,276 images; ≈ 10 s per
evaluation on one GPU):

| Checkpoint | mAP50 (VOC07) |
|---|---|
| all-DCNM INT8 QAT | 0.8892 |
| HERA-A (scheme7) | 0.8859 |
| HERA-P (scheme9) | 0.8512 |
| all-ACIM | 0.5428 |

The CPU mapping demo for this workload is `python examples/fasterrcnn/reproduce_mapping.py
--profiles-json examples/data/fasterrcnn_synthetic_profile.json`. The all-DCNM evaluation is
deterministic; hybrid configurations sample ACIM noise at evaluation time, so their mAP
varies slightly across seeds.

### PrivateLoRA (cloud-edge collaborative LLM)

`examples/privatelora/` contains the PrivateLoRA Llama-2-7b model definition (96 PLM
`lora_mobile` matrices + Head LB), the INT8/CGRA quantization adapter, and a bundled
lm-evaluation-harness fork for BoolQ / GSM8K evaluation.

```bash
cd examples/privatelora
export PRIVATE_LLM_BASE_MODEL=/path/to/llama-2-7b-hf   # or the HF Hub id
# checkpoints: see weights/README.md (v1.0.0 release); BoolQ/GSM8K auto-fetch from HF Hub
python eval_pl.py --checkpoint-path /path/to/pl_boolq_fp_r256.bin --lora-rank 256     --tasks boolq --batch-size 16 --output-json results/boolq_fp.json
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
that VGG16 (real profiles) and Faster R-CNN reproduce the manuscript's mapping splits. They
run in under 1 second.

## Citation

Citation information will be added here upon publication.

## Contact

For questions regarding the code or the manuscript, please contact the corresponding authors
listed in the paper.
