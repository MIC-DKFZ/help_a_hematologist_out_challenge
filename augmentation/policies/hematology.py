import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.color_transforms import (
    ContrastAugmentationTransform,
    BrightnessTransform,
    GammaTransform,
)
from batchgenerators.transforms.local_transforms import (
    BrightnessGradientAdditiveTransform,
    LocalGammaTransform,
    LocalContrastTransform,
    LocalSmoothingTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
    SharpeningTransform,
    MedianFilterTransform,
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    TransposeAxesTransform,
    SpatialTransform_2,
    ResizeTransform,
)
from batchgenerators.transforms.utility_transforms import OneOfTransformPerSample, NumpyToTensor

from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import cv2
from skimage import io
import tifffile
from typing import Tuple, Callable, Union
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from acvl_utils.cropping_and_padding.padding import pad_nd_image

from typing import List, Tuple, Optional, Dict
from torch import Tensor
import math
from torchvision.transforms import functional as F, InterpolationMode
import random

cv2.setNumThreads(1)

intensity_stats = {
    "Matek_19": (
        np.array([209.32721942, 185.69440888, 213.27916735]),
        np.array([223.0, 205.0, 222.0]),
        np.array([43.65395526, 65.93844005, 26.23027092]),
    ),
    "Acevedo_20": (
        np.array([221.61488762, 189.0902095, 182.58819691]),
        np.array([237.0, 208.0, 188.0]),
        np.array([42.26784376, 49.32332775, 20.42777253]),
    ),
    "WBC1": (
        np.array([188.65644123, 166.30655814, 198.53313451]),
        np.array([202.0, 178.0, 211.0]),
        np.array([52.63319061, 68.50900374, 45.91395474]),
    ),
}


def get_bg_train_transform(data_dir):

    patch_size = (288, 288)

    percentage_darkest_pixels = 20

    hsv_wbc = collect_wbc_hsv(data_dir, percentage_darkest_pixels)

    tr_transforms = TorchCompose(
        [
            OneOfTransformPerSample(
                [
                    RandomHueTransform(
                        [i[0] for i in hsv_wbc], p_per_sample=0.5, percentage_darkest_pixels=percentage_darkest_pixels
                    ),
                    HueJitterTransform(factor=(-0.1, 0.1), p_per_sample=0.5),
                ],
                relevant_keys=["data"],
            ),
            SpatialTransform_2(
                patch_size,
                [(i // 2 + 30) for i in patch_size],
                do_elastic_deform=True,
                deformation_scale=(0, 0.25),
                do_rotation=True,  # default is full rotations
                do_scale=True,
                scale=(0.5, 2),
                border_mode_data="constant",
                order_data=1,
                random_crop=True,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.5,
                p_rot_per_sample=0.5,
                independent_scale_for_each_axis=True,
                p_independent_scale_per_axis=0.3,
            ),
            MirrorTransform((0, 1)),
            TransposeAxesTransform((0, 1)),
            # global brightness/contrast transforms
            OneOfTransformPerSample(
                [
                    BrightnessTransform(mu=0, sigma=10, per_channel=False, p_per_sample=0.5),
                    ContrastAugmentationTransform(
                        (0.75, 1.25), preserve_range=False, per_channel=False, p_per_sample=0.5
                    ),
                    GammaTransform((0.5, 2), invert_image=False, per_channel=False, p_per_sample=0.5),
                    GammaTransform((0.5, 2), invert_image=True, per_channel=False, p_per_sample=0.5),
                ],
                relevant_keys=["data"],
                p=(0.33, 0.33, 0.17, 0.17),
            ),
            # blur and sharpening transforms
            OneOfTransformPerSample(
                [
                    LocalSmoothingTransform(
                        lambda x, y: np.random.uniform(x[y] // 8, x[y] // 2),
                        (-0.2, 1.2),
                        smoothing_strength=(0.5, 1),
                        kernel_size=(0.3, 3),
                        same_for_all_channels=True,
                        p_per_sample=0.5,
                        p_per_channel=1,
                    ),
                    MedianFilterTransform((1, 3), same_for_each_channel=True, p_per_sample=0.5, p_per_channel=1),
                    SimulateLowResolutionTransform(
                        zoom_range=(0.3, 1), per_channel=False, order_downsample=0, order_upsample=1, p_per_sample=0.5
                    ),
                    GaussianBlurTransform(
                        (0.5, 1.0),
                        different_sigma_per_channel=False,
                        p_per_sample=0.5,
                        p_per_channel=1,
                        different_sigma_per_axis=True,
                        p_isotropic=0.3,
                    ),
                ],
                relevant_keys=["data"],
                p=(0.4, 0.2, 0.2, 0.2),
            ),
            # local intensity/contrast
            OneOfTransformPerSample(
                [
                    LocalContrastTransform(
                        lambda x, y: np.random.uniform(x[y] // 8, x[y] // 2),
                        (-0.2, 1.2),
                        new_contrast=(1e-5, 2),
                        same_for_all_channels=True,
                        p_per_sample=0.5,
                        p_per_channel=1,
                    ),
                    LocalGammaTransform(
                        lambda x, y: np.random.uniform(x[y] // 8, x[y] // 2),
                        (-0.2, 1.2),
                        lambda: np.random.uniform(0.1, 0.8) if np.random.uniform() < 0.5 else np.random.uniform(1.5, 4),
                        same_for_all_channels=True,
                        p_per_sample=0.5,
                        p_per_channel=1,
                    ),
                    BrightnessGradientAdditiveTransform(
                        lambda x, y: np.random.uniform(x[y] // 8, x[y] // 2),
                        (-0.2, 1.2),
                        max_strength=lambda x, y: np.random.uniform(-30, 30),
                        same_for_all_channels=True,
                        p_per_sample=0.5,
                        p_per_channel=1,
                    ),
                ],
                relevant_keys=["data"],
                p=(0.33, 0.27, 0.4),
            ),
            # noise Transforms
            SharpeningTransform(strength=(0.1, 1), same_for_each_channel=True, p_per_sample=0.2, p_per_channel=1),
            GaussianNoiseTransform(noise_variance=(1e-5, 10), p_per_sample=0.15, p_per_channel=1, per_channel=False),
            MeanNormalizationTransform(intensity_stats["WBC1"][0]),
            # NormalizeTransform(intensity_stats['WBC1'][0], intensity_stats['WBC1'][2], 'data'),
            # BlankRectangleTransform([[max(1, p // 10), p // 3] for p in self.patch_size],
            #                         rectangle_value=0,
            #                         num_rectangles=(1, 3.01),
            #                         force_square=False,
            #                         p_per_sample=0.25,
            #                         p_per_channel=1
            #                         ),
            NumpyToTensor(keys=["data"], cast_to="float"),
            # NumpyToTensor(keys=["target"], cast_to="long"),
        ]
    )

    return tr_transforms


def get_bg_val_transform():
    val_transforms = TorchCompose(
        [
            ResizeTransform([288, 288]),
            MeanNormalizationTransform(intensity_stats["WBC1"][0]),
            # NormalizeTransform(intensity_stats['WBC1'][0], intensity_stats['WBC1'][2], 'data'),
            NumpyToTensor(keys=["data"], cast_to="float"),
            # NumpyToTensor(keys=['target'], cast_to='long'),
        ]
    )

    return val_transforms


class TorchCompose(Compose):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __call__(self, x, mask=None):

        # transform to numpy if needed
        if not isinstance(x, np.ndarray):
            if isinstance(x, torch.Tensor):
                x = x.numpy()
            elif isinstance(x, Image.Image):
                x = np.array(x, dtype=np.float32).transpose((2, 0, 1))

        x = x.copy()

        # add channel dim in case a grayscale img doesn't have one
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)

        # expand dim since batchgenerators expects a batch dim
        x = np.expand_dims(x, 0)
        if mask:
            mask = np.expand_dims(mask, 0)

        data_dict = {"data": x, "seg": mask}

        for t in self.transforms:
            data_dict = t(**data_dict)

        # extract data from dict and squeeze batch dim
        x = data_dict["data"].squeeze(axis=0)
        if mask:
            mask = data_dict["seg"].squeeze(axis=0)

        if mask:
            return x, mask
        else:
            return x


class HueJitterTransform(AbstractTransform):
    def __init__(self, factor: Union[Tuple[float, float], Callable[[], float]], p_per_sample: float = 1):
        self.factor = factor
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                image = data[b].transpose((1, 2, 0)).astype(np.float32)  # x, y, 3
                image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                if isinstance(self.factor, tuple):
                    factor = np.random.uniform(*self.factor)
                else:
                    factor = self.factor()
                image_hsv[..., 0] = np.mod(image_hsv[..., 0] + factor * 360, 360)
                image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
                image = image.transpose((2, 0, 1))
                data[b] = image
        data_dict["data"] = data
        return data_dict


class RandomHueTransform(AbstractTransform):
    def __init__(self, reference_hues: List[float], p_per_sample: float = 1, percentage_darkest_pixels=20):
        self.reference_hues = reference_hues
        self.p_per_sample = p_per_sample
        self.percentage_darkest_pixels = percentage_darkest_pixels

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                target_hue = self.reference_hues[np.random.choice(len(self.reference_hues))]

                image = data[b].transpose((1, 2, 0)).astype(np.float32)  # x, y, 3
                image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                luminances = image_hsv[:, :, 2]
                luminances = luminances[luminances > 0]

                mask = (image_hsv[:, :, 2] < np.percentile(luminances, self.percentage_darkest_pixels)) & (
                    image_hsv[:, :, 2] > 0
                )
                mean_luminance = image_hsv[mask][:, 0].mean()

                image_hsv[:, :, 0] = np.mod(image_hsv[:, :, 0] - mean_luminance + target_hue, 360)

                image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
                image = image.transpose((2, 0, 1))
                data[b] = image

        data_dict["data"] = data
        return data_dict


class MeanNormalizationTransform(AbstractTransform):
    def __init__(self, mean_values: np.ndarray):
        self.mean_values = mean_values

    def __call__(self, **data_dict):
        data_dict["data"] -= self.mean_values[None, :, None, None]
        return data_dict


def load_image(image_fname: str, crop_size: Tuple[int, int] = None):
    if image_fname.lower().endswith(".tiff"):
        image = tifffile.imread(image_fname)
    else:
        image = io.imread(image_fname)
    assert image.shape[-1] == 3 or image.shape[-1] == 4
    if image.shape[-1] == 4:
        image = image[:, :, :-1]
    image = image.transpose((2, 0, 1))
    if crop_size is not None:
        image = pad_nd_image(image, new_shape=crop_size, mode="constant")
        image = crop(image[None], crop_size=crop_size)[0][0]
    return image


def collect_intensity_properties_image_hsv(image: str, num_samples: int = 1000):
    image_npy = load_image(image_fname=image)
    image_npy = image_npy.transpose((1, 2, 0)).astype(np.float32)
    image_npy = cv2.cvtColor(image_npy, cv2.COLOR_RGB2HSV)
    idx = np.random.choice(len(image_npy[:, :, 0].ravel()), size=num_samples)
    return [image_npy[:, :, i].flatten()[idx] for i in range(3)]


def collect_wbc_hsv(hemato_base, percentage_darkest_pixels=10):
    list_of_images = subfiles(join(hemato_base, "WBC1/DATA-VAL"), join=True, suffix=".TIF")
    samples_per_image = 5000
    p = Pool(12)
    res = p.starmap(
        collect_intensity_properties_image_hsv, zip(list_of_images, [samples_per_image] * len(list_of_images))
    )
    p.close()
    p.join()
    # we only take the darkest pixels
    r2 = []
    for r in res:
        luminances = r[2]
        luminances = luminances[luminances > 0]
        cutoff = np.percentile(luminances, percentage_darkest_pixels)
        mask = (luminances < cutoff) & (luminances > 0)
        r2.append((r[0][mask], r[1][mask], r[2][mask]))
    tmp = [np.mean(i, 1) for i in r2]
    # save_pickle(tmp, join(hemato_base, 'wbc_hsv.pkl'))
    return tmp


def get_starter_train():

    resize = 224
    random_crop_scale = (0.8, 1.0)
    random_crop_ratio = (0.8, 1.2)

    mean = [0.485, 0.456, 0.406]  # values from imagenet
    std = [0.229, 0.224, 0.225]  # values from imagenet

    normalization = torchvision.transforms.Normalize(mean, std)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(resize, scale=random_crop_scale, ratio=random_crop_ratio),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalization,
        ]
    )

    return train_transform


def get_starter_test():

    resize = 224
    mean = [0.485, 0.456, 0.406]  # values from imagenet
    std = [0.229, 0.224, 0.225]  # values from imagenet

    normalization = torchvision.transforms.Normalize(mean, std)

    test_transform = transforms.Compose([transforms.Resize([resize, resize]), transforms.ToTensor(), normalization])

    return test_transform


def old_get_starter_train():

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
        ]
    )

    return train_transform


def old_get_starter_test():

    resize = 224
    mean = [0.485, 0.456, 0.406]  # values from imagenet
    std = [0.229, 0.224, 0.225]  # values from imagenet

    normalization = torchvision.transforms.Normalize(mean, std)

    test_transform = transforms.Compose([normalization, transforms.Resize(resize)])

    return test_transform


### Custom Randaugment by Lars ###
def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    elif op_name == "Hue":
        img = F.adjust_hue(img, magnitude)
    elif op_name == "Saturation":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "HSV":
        img = F.adjust_hue(img, magnitude)
        img = F.adjust_saturation(img, 1.0 + magnitude)
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


class customRandAugment(torchvision.transforms.RandAugment):
    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        random_range=True,
    ) -> None:
        super().__init__()
        self.random_range = random_range

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            # "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            # "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            # "Equalize": (torch.tensor(0.0), False),
            "Hue": (torch.linspace(0.0, 0.25, num_bins), True),
            "Saturation": (torch.linspace(0.0, 0.9, num_bins), True),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            if self.random_range:
                magnitude = random.uniform(0, magnitude)
            # print(op_name, magnitude)
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img


def get_customRandAugment():
    resize = 224
    random_crop_scale = (0.8, 1.0)
    random_crop_ratio = (0.8, 1.2)

    mean = [0.485, 0.456, 0.406]  # values from imagenet
    std = [0.229, 0.224, 0.225]  # values from imagenet

    normalization = torchvision.transforms.Normalize(mean, std)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(resize, scale=random_crop_scale, ratio=random_crop_ratio),
            customRandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalization,
        ]
    )

    return train_transform
