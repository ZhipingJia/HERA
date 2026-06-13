"""VGG models from chenyaofo/pytorch-cifar-models (BSD-3-Clause).

Only the VGG variants are vendored here; ``cifar100_vgg16_bn(pretrained=True)``
auto-downloads the public CIFAR-100 weights from the upstream GitHub release.
"""

from .vgg import cifar10_vgg11_bn, cifar10_vgg13_bn, cifar10_vgg16_bn, cifar10_vgg19_bn
from .vgg import cifar100_vgg11_bn, cifar100_vgg13_bn, cifar100_vgg16_bn, cifar100_vgg19_bn
