from dataclasses import dataclass
from pathlib import Path

import torchvision
import torchvision.transforms as transforms


@dataclass(frozen=True)
class VGG16DatasetProfile:
    dataset: str
    model_name: str
    num_classes: int
    weights_path: str
    quant_best_path: str
    data_root: str = './data'

    @property
    def dataset_cls(self):
        return torchvision.datasets.CIFAR10 if self.dataset == 'cifar10' else torchvision.datasets.CIFAR100


PROFILES = {
    'cifar10': VGG16DatasetProfile(
        dataset='cifar10',
        model_name='cifar10_vgg16_bn',
        num_classes=10,
        weights_path='',  # FP weights via pytorch_cifar_models(pretrained=True)
        quant_best_path='',  # INT8 baseline via --int8-checkpoint (released asset)
    ),
    'cifar100': VGG16DatasetProfile(
        dataset='cifar100',
        model_name='cifar100_vgg16_bn',
        num_classes=100,
        weights_path='',  # FP weights via pytorch_cifar_models(pretrained=True)
        quant_best_path='',  # INT8 baseline via --int8-checkpoint (released asset)
    ),
}


def get_profile(dataset: str) -> VGG16DatasetProfile:
    if dataset not in PROFILES:
        raise ValueError(f'Unsupported dataset: {dataset}')
    return PROFILES[dataset]



def build_cifar_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
