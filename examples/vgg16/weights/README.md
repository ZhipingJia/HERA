# VGG16 Checkpoints

The affinity framework's audit network uses two checkpoints:

| Role | Source |
|---|---|
| Full-precision VGG16-BN (CIFAR-100) | Public [chenyaofo](https://github.com/chenyaofo/pytorch-cifar-models) weights — auto-downloaded by `pytorch_cifar_models`; **not redistributed here**. |
| DCNM INT8 baseline (KLD-profiling reference) | `vgg16_int8_baseline_cifar100.pth` — published as a GitHub Release asset (~59 MB). |

The INT8 baseline is the reference model for single-layer ACIM substitution in
`profile_kld.py` and the INT8 accuracy in `eval.py --mode int8`. It is released at:

https://github.com/ZhipingJia/HERA/releases

After download, place it here as `weights/vgg16_int8_baseline_cifar100.pth` (or pass
`--int8-checkpoint`). The full-precision weights need no manual download — `eval.py`,
`profile_edp.py`, and `profile_kld.py` obtain them through `cifar100_vgg16_bn(pretrained=True)`.

CIFAR-100 itself is fetched by torchvision into `--data-root` (default `./cifar_data`).
