# Faster R-CNN Checkpoints

All detector checkpoints for the lightweight channel-32 Faster R-CNN are small
(~320 KB each) and ship directly in this directory. Each file is a sanitized
`{"model": state_dict, "meta": {...}}` payload.

| File | Configuration | mAP50 (VOC07) |
|---|---|---|
| `fasterrcnn_all_dcnm_int8_qat.pth` | All target layers on DCNM (INT8 QAT); reference model for KLD profiling | 0.8892 |
| `fasterrcnn_hera_a_scheme7.pth` | HERA-A: extractor convs + rpn.conv1 on ACIM, ROI head on DCNM | 0.8859 |
| `fasterrcnn_hera_p_scheme9.pth` | HERA-P: HERA-A + ROI classifier FC1/FC2 on ACIM | 0.8512 |
| `fasterrcnn_all_acim.pth` | All 14 target layers on ACIM | 0.5428 |
| `fasterrcnn_fp.pth` | Full-precision initialization for `build_dcnm_baseline.py` (see note) | — |

Evaluate the all-DCNM model with `--quant-config configs/config_dcnm_int8.yaml`;
evaluate the hybrid models with the matching
`configs/effective_config_*.json` via `--effective-config`. ACIM layers inject
sampled noise at evaluation time, so hybrid mAP varies slightly across seeds
(the all-DCNM value is deterministic and reproduces exactly).

Note on the full-precision checkpoint: its recorded accuracy was measured under
the original digital INT8-emulation evaluation flow, which is not part of this
repository; a plain full-precision forward pass does not reproduce it. Its role
here is as the FP starting point for the DCNM INT8 PTQ/QAT pipeline.

The thermal human-detection dataset (UNIRI-TID) is not distributed; see the repository
README for the expected VOC-style layout (1280×960 frames, single class `human`, fixed
`valid_crop`) to run on your own data.
