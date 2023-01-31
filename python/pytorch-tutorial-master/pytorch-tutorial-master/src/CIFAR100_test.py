

from torchvision.datasets import CIFAR10, CIFAR100

image_datasets = CIFAR100("../data", train=True, download=True)

print()