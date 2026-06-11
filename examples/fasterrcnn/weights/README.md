# Faster R-CNN Checkpoints

All detector checkpoints for the lightweight channel-32 Faster R-CNN are small
(~320 KB each) and ship directly in this directory. Each file is a sanitized
`{"model": state_dict, "meta": {...}}` payload.

| File | Configuration | mAP50 (VOC07) |
|---|---|---|
| `fasterrcnn_all_dcnm_int8_qat.pth` | All target layers on DCNM (INT8 QAT); reference model for KLD profiling | 0.8947 |
| `fasterrcnn_hera_a_scheme7_noise20.pth` | HERA-A: extractor convs + rpn.conv1 on ACIM, ROI head on DCNM | 0.8814 |
| `fasterrcnn_hera_p_scheme9_noise20.pth` | HERA-P: HERA-A + ROI classifier FC1/FC2 on ACIM | 0.8462 |
| `fasterrcnn_all_acim_noise20.pth` | All 14 target layers on ACIM | 0.5429 |
| `fasterrcnn_fp_map0889_epoch10.pth` | Full-precision initialization for `build_dcnm_baseline.py` (see note) | — |

Evaluate the all-DCNM model with `--quant-config configs/config_dcnm_int8.yaml`;
evaluate the hybrid models with the matching
`configs/effective_config_*.json` via `--effective-config`. ACIM layers inject
sampled noise at evaluation time, so hybrid mAP reproduces within roughly
±0.002 of the values above (the all-DCNM value is deterministic and reproduces
exactly).

Note on the full-precision checkpoint: its 0.8889 mAP50 (in the filename) was
measured under the original digital INT8-emulation evaluation flow, which is
not part of this repository; a plain full-precision forward pass does not
reproduce that number. Its role here is as the FP starting point for the DCNM
INT8 PTQ/QAT pipeline.

The infrared human-detection dataset is not distributed; see the repository
README for the expected VOC-style layout to run on your own data.
