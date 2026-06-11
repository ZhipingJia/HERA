# Synthetic Demo Profiles

This directory contains synthetic profile JSON files for demonstrating the HERA
mapping-report scripts. Each file covers the full layer set of its workload
(Faster R-CNN: 14 layers; PrivateLoRA: 96 PLM matrices + Head LB; VGG16: 16
layers).

The numbers are illustrative placeholders only. They are not paper-reported
measurements, not real HERA power/latency data, and not derived from private
datasets, checkpoints, or training logs. They are, however, constructed so that
the affinity framework reproduces the qualitative HERA-A / HERA-P split each
workload reports in the manuscript (e.g. Faster R-CNN keeps the ROI-head
classifier layers on DCNM under HERA-A and moves them to ACIM under HERA-P).

Each profile row uses the minimal schema consumed by the example scripts:

- `name`: full layer name
- `short_name`: compact display name
- `kld`: illustrative per-layer KLD
- `edp_acim`: illustrative ACIM EDP
- `edp_dcnm`: illustrative DCNM EDP

