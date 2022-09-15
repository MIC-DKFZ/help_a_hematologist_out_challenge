import torch
import torchvision
from torchvision import transforms


def get_starter_train():

    resize = 224
    random_crop_scale = (0.8, 1.0)
    random_crop_ratio = (0.8, 1.2)

    mean = [0.485, 0.456, 0.406]  # values from imagenet
    std = [0.229, 0.224, 0.225]  # values from imagenet

    normalization = torchvision.transforms.Normalize(mean, std)

    train_transform = transforms.Compose(
        [
            normalization,
            transforms.RandomResizedCrop(resize, scale=random_crop_scale, ratio=random_crop_ratio),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ToTensor(),
        ]
    )

    return train_transform


def get_starter_test():

    resize = 224
    mean = [0.485, 0.456, 0.406]  # values from imagenet
    std = [0.229, 0.224, 0.225]  # values from imagenet

    normalization = torchvision.transforms.Normalize(mean, std)

    test_transform = transforms.Compose([normalization, transforms.Resize(resize)])

    return test_transform
