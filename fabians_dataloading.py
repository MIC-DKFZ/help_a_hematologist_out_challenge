import os.path

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
from batchgenerators.transforms.spatial_transforms import MirrorTransform, TransposeAxesTransform, SpatialTransform_2
from batchgenerators.transforms.utility_transforms import OneOfTransformPerSample, NumpyToTensor

from augmentation.policies.hematology import (
    MeanNormalizationTransform,
    RandomHueTransform,
    HueJitterTransform,
    collect_wbc_hsv,
    intensity_stats,
)

import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading.data_loader import DataLoader
from skimage import io
import tifffile

from typing import Tuple, Callable
from batchgenerators.utilities.file_and_folder_operations import *

class_name_to_id = {
    "basophil": 0,
    "eosinophil": 1,
    "erythroblast": 2,
    "lymphocyte_typical": 3,
    "metamyelocyte": 4,
    "monocyte": 5,
    "myelocyte": 6,
    "neutrophil_banded": 7,
    "neutrophil_segmented": 8,
    "promyelocyte": 9,
    "myeloblast": 10,
}


class HelpAHematoOutDatasetVal(DataLoader):
    def __init__(
        self,
        data,
        normalization_fn: Callable,
        patch_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        infinite: bool = False,
        num_threads_in_multithreaded: int = 1,
    ):
        # that's retarded. We should just give this as part of data
        dataset_ids = [i.split(os.path.sep)[-3] for i in data[0]]

        super().__init__(
            (*data, dataset_ids),
            batch_size,
            num_threads_in_multithreaded=num_threads_in_multithreaded,
            seed_for_shuffle=None,
            return_incomplete=True,
            shuffle=True,
            infinite=infinite,
            sampling_probabilities=None,
        )
        self.patch_size = patch_size
        self.normalization_fn = normalization_fn
        self.indices = list(range(len(data[0])))

    def generate_train_batch(self):
        indices = self.get_indices()
        batch = np.zeros((len(indices), 3, *self.patch_size), dtype=np.float32)
        labels = np.zeros(len(indices), dtype=int)
        source_files = []
        for i, ind in enumerate(indices):
            source_files.append(self._data[0][ind])
            batch[i] = self.normalization_fn(load_image(self._data[0][ind], self.patch_size), self._data[-1][ind])
            labels[i] = self._data[1][ind]
        return {"data": batch, "target": labels, "source_files": source_files}


class HelpAHematoOutDatasetTrain(DataLoader):
    def __init__(
        self,
        data,
        normalization_fn: Callable,
        patch_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        classes: int = 11,
        oversample_percent: float = 0.33,
    ):
        # let's infer the images associated with each source dataset and store them separately
        # that's retarded. We should just give this as part of data
        dataset_ids = [i.split(os.path.sep)[-3] for i in data[0]]
        unique_datasets = np.unique(dataset_ids)
        data_by_dataset = {u: ([], [], {c: [] for c in range(classes)}) for u in unique_datasets}
        for i, d in enumerate(dataset_ids):
            data_by_dataset[d][0].append(data[0][i])
            data_by_dataset[d][1].append(data[1][i])
            data_by_dataset[d][2][data[1][i]].append(len(data_by_dataset[d][0]) - 1)  # confused yet?

        for d in data_by_dataset.keys():
            for k in list(data_by_dataset[d][2].keys()):
                if len(data_by_dataset[d][2][k]) == 0:
                    del data_by_dataset[d][2][k]

        self.oversample_percent = oversample_percent
        self.patch_size = patch_size
        self.normalization_fn = normalization_fn
        self.indices = list(range(len(data[0])))

        super().__init__(
            data_by_dataset,
            batch_size,
            infinite=True,
            num_threads_in_multithreaded=1,
            seed_for_shuffle=None,
            return_incomplete=True,
            shuffle=True,
            sampling_probabilities=None,
        )

    def get_indices(self):
        # for 33% of samples pick a dataset, then a class, then a sample
        idx = []
        for i in range(self.batch_size):
            dataset = np.random.choice(list(self._data.keys()))
            if np.random.uniform() > self.oversample_percent:
                # pick a random case
                sample = np.random.choice(len(self._data[dataset][0]))
            else:
                # pick a random class, then a random case
                my_picked_class = np.random.choice(list(self._data[dataset][2].keys()))
                sample = np.random.choice(self._data[dataset][2][my_picked_class])
                assert self._data[dataset][1][sample] == my_picked_class
            idx.append((dataset, sample))
        return idx

    def generate_train_batch(self):
        batch = np.zeros((self.batch_size, 3, *self.patch_size), dtype=np.float32)
        labels = np.zeros(self.batch_size, dtype=int)
        source_files = []
        for i, (dataset, sample) in enumerate(self.get_indices()):
            source_files.append(self._data[dataset][0][sample])
            batch[i] = self.normalization_fn(load_image(self._data[dataset][0][sample], self.patch_size), dataset)
            labels[i] = self._data[dataset][1][sample]
        return {"data": batch, "target": labels, "source_files": source_files}


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


def load_dataset_from_folder(base_folder: str):
    subfolders = class_name_to_id.keys()
    images = []
    labels = []
    for l in subfolders:
        label_dir = join(base_folder, l)
        if isdir(label_dir):
            image_files = [
                i
                for i in subfiles(label_dir, join=True)
                if i.lower().endswith(".jpg")
                or i.lower().endswith(".tif")
                or i.lower().endswith(".tiff")
                or i.lower().endswith(".png")
            ]
            labels += [class_name_to_id[l]] * len(image_files)
            images += image_files
    return images, labels


def get_dataloaders_fabian(data_dir, patch_size, initial_patch_size, batch_size, num_processes_DA, train_setting):
    data_acevedo, labels_acevedo = load_dataset_from_folder(join(data_dir, "Acevedo_20_meanNormalized"))
    data_matek, labels_matek = load_dataset_from_folder(join(data_dir, "Matek_19_meanNormalized"))

    if train_setting == "matek_to_acevedo":
        train_data, train_labels = data_matek, labels_matek
        val_data, val_labels = data_acevedo, labels_acevedo
        num_classes = 10
        # remove class 10
        train_data = [train_data[i] for i in range(len(train_data)) if train_labels[i] != 10]
        train_labels = [train_labels[i] for i in range(len(train_labels)) if train_labels[i] != 10]
        val_data = [val_data[i] for i in range(len(val_data)) if val_labels[i] != 10]
        val_labels = [val_labels[i] for i in range(len(val_labels)) if val_labels[i] != 10]

    elif train_setting == "acevedo_to_matek":
        train_data, train_labels = data_acevedo, labels_acevedo
        val_data, val_labels = data_matek, labels_matek
        num_classes = 10
        # remove class 10
        train_data = [train_data[i] for i in range(len(train_data)) if train_labels[i] != 10]
        train_labels = [train_labels[i] for i in range(len(train_labels)) if train_labels[i] != 10]
        val_data = [val_data[i] for i in range(len(val_data)) if val_labels[i] != 10]
        val_labels = [val_labels[i] for i in range(len(val_labels)) if val_labels[i] != 10]

    elif train_setting.startswith("fold_"):
        fold_id = int(train_setting[-1])
        num_classes = 11
        splits = load_json(join(data_dir, "splits.json"))
        train_data = []
        train_labels = []
        for tr in splits[fold_id]["train"]:
            dataset, class_name, image = tr.split(os.path.sep)
            train_data.append(join(data_dir, dataset + "_meanNormalized", class_name, image))
            train_labels.append(class_name_to_id[class_name])
        val_data = []
        val_labels = []
        for val in splits[fold_id]["val"]:
            dataset, class_name, image = val.split(os.path.sep)
            val_data.append(join(data_dir, dataset + "_meanNormalized", class_name, image))
            val_labels.append(class_name_to_id[class_name])
    else:
        raise RuntimeError()

    length_train = len(train_data) // batch_size
    length_val = len(val_data) // batch_size

    tr_gen = HelpAHematoOutDatasetTrain((train_data, train_labels), lambda x, y: x, initial_patch_size, batch_size)
    val_gen = HelpAHematoOutDatasetVal(
        (val_data, val_labels),
        lambda x, y: x,
        patch_size,
        batch_size,
        num_threads_in_multithreaded=num_processes_DA // 2,
    )

    percentage_darkest_pixels = 20
    # this runs in like 1 second so it's fine to do this everytime
    hsv_wbc = collect_wbc_hsv(data_dir, percentage_darkest_pixels)

    tr_transforms = Compose(
        [
            # we need to apply hue jitter first because we need to have black pixels where we padded. Otherwise it's
            # good night
            OneOfTransformPerSample(
                [
                    RandomHueTransform(
                        [i[0] for i in hsv_wbc], p_per_sample=0.5, percentage_darkest_pixels=percentage_darkest_pixels
                    ),
                    HueJitterTransform(factor=(-0.1, 0.1), p_per_sample=0.5),
                ],
                relevant_keys=["data"],
            ),
            # das ist mal ein Brett. Mach das zuerst junge sonst haste nen CPU bottleneck wa
            SpatialTransform_2(
                patch_size,
                [(i // 2 + 30) for i in patch_size],
                do_elastic_deform=True,
                deformation_scale=(0, 0.25),
                do_rotation=True,  # default is full rotations
                do_scale=True,
                scale=(0.5, 2),  # need to go overboard with this I think
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
            NumpyToTensor(keys=["target"], cast_to="long"),
        ]
    )
    val_transforms = Compose(
        [
            MeanNormalizationTransform(intensity_stats["WBC1"][0]),
            # NormalizeTransform(intensity_stats['WBC1'][0], intensity_stats['WBC1'][2], 'data'),
            NumpyToTensor(keys=["data"], cast_to="float"),
            NumpyToTensor(keys=["target"], cast_to="long"),
        ]
    )

    tr_gen_mt = NonDetMTAWithLength(
        tr_gen, tr_transforms, num_processes_DA, length=length_train, num_cached=64, pin_memory=True
    )
    val_gen_mt = MTAWithLength(
        val_gen, val_transforms, num_processes_DA // 2, length=length_val, num_cached_per_queue=2, pin_memory=True
    )
    return tr_gen_mt, val_gen_mt


class MTAWithLength(MultiThreadedAugmenter):
    def __init__(
        self,
        data_loader,
        transform,
        num_processes,
        length,
        num_cached_per_queue=2,
        seeds=None,
        pin_memory=False,
        timeout=10,
        wait_time=0.02,
    ):
        super().__init__(
            data_loader, transform, num_processes, num_cached_per_queue, seeds, pin_memory, timeout, wait_time
        )
        self.length = length

    def __len__(self):
        return self.length


class NonDetMTAWithLength(NonDetMultiThreadedAugmenter):
    def __init__(
        self,
        data_loader,
        transform,
        num_processes,
        length,
        num_cached=2,
        seeds=None,
        pin_memory=False,
        wait_time=0.02,
    ):
        super().__init__(data_loader, transform, num_processes, num_cached, seeds, pin_memory, wait_time)
        self.length = length

    def __len__(self):
        return self.length
