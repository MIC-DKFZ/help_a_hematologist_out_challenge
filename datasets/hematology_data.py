import torch
from torch.utils.data import Dataset, WeightedRandomSampler, Sampler
from imageio import imread
import numpy as np
import copy
import os
import glob
import json
from PIL import Image
import random
from typing import Iterator


crop_Ace20 = 250
crop_Mat19 = 345
crop_WBC1 = 288

dataset_image_size = {
    "Acevedo_20": crop_Ace20,  # 250,
    "Matek_19": crop_Mat19,  # 345,
    "WBC1": crop_WBC1,  # 288,
}


class HematologyDataset(Dataset):
    def __init__(self, data_dir, dset, train=True, transform=None, split_file=None, starter_crops=False):
        """
        data_dir: Path to parent_dir where the 3 dataset folders are located
        dset: "acevedo" (train on acevedo, val on matek), "matek" (train on matek, val on acevedo), "combined" (20/80 split on full data)
        train: True if train, False if validation
        """
        self.transform = transform
        self.starter_crops = starter_crops

        label_map = {
            "basophil": 0,
            "eosinophil": 1,
            "erythroblast": 2,
            "myeloblast": 3,
            "promyelocyte": 4,
            "myelocyte": 5,
            "metamyelocyte": 6,
            "neutrophil_banded": 7,
            "neutrophil_segmented": 8,
            "monocyte": 9,
            "lymphocyte_typical": 10,
        }

        acevedo_dir = os.path.join(data_dir, "Acevedo_20")
        matek_dir = os.path.join(data_dir, "Matek_19")

        if dset == "acevedo":

            if train:
                self.files = glob.glob(os.path.join(acevedo_dir, "*/*.jpg"))
            else:
                self.files = glob.glob(os.path.join(matek_dir, "*/*.tiff"))

        elif dset == "matek":

            if train:
                self.files = glob.glob(os.path.join(matek_dir, "*/*.tiff"))
            else:
                self.files = glob.glob(os.path.join(acevedo_dir, "*/*.jpg"))

        elif dset == "combined":

            f = open(split_file)
            json_data = json.load(f)
            train_files = json_data[0]["train"]  # fold 0 only for now
            val_files = json_data[0]["val"]  # fold 0 only for now

            if train:
                self.files = [os.path.join(data_dir, i) for i in train_files]
            else:
                self.files = [os.path.join(data_dir, i) for i in val_files]

        self.labels = [label_map[i.split("/")[-2]] for i in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # img = (torch.from_numpy(np.array(imread(self.files[idx])[:, :, :3])) / 255.0).permute(2, 0, 1)

        image = imread(self.files[idx])[:, :, :3]

        if self.starter_crops:
            # print("before", image.shape)
            orginal_dataset = self.files[idx].split("/")[-3]
            crop_size = dataset_image_size[orginal_dataset]
            h1 = (image.shape[0] - crop_size) / 2
            h1 = int(h1)
            h2 = (image.shape[0] + crop_size) / 2
            h2 = int(h2)

            w1 = (image.shape[1] - crop_size) / 2
            w1 = int(w1)
            w2 = (image.shape[1] + crop_size) / 2
            w2 = int(w2)
            image = image[h1:h2, w1:w2, :]
            # print("after", image.shape)

        img = Image.fromarray(image)

        if self.transform:
            img = self.transform(img)

        lb = self.labels[idx]

        return img, lb

    def get_weighted_random_sampler(self, balanced=False):

        # WeightedRandomSampler: https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
        if balanced:
            # class_sample_count = np.array([len(np.where(self.labels == t)[0]) for t in np.unique(self.labels)])
            class_sample_count = np.array([len(np.where(self.labels == np.int16(t))[0]) for t in list(range(11))])
            weight = 1.0 / class_sample_count
            weight[weight == np.inf] = 0
        else:
            weight = [0.03, 0.97]  # TODO insert weights for specific classes
        samples_weight = np.array([weight[t] for t in self.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler


class SamplerCombiner(Sampler[int]):
    def __init__(self, sampler_a, sampler_b, prob_a=2 / 3):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        self.prob_a = prob_a

        self.num_samples = sampler_a.num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        self.sampler_a = list(self.sampler_a.__iter__())
        self.sampler_b = list(self.sampler_b.__iter__())
        self.index = 0
        return self

    def __next__(self):

        if self.index < len(self):

            if np.random.rand() <= self.prob_a:
                # sampler_a will be used
                out = random.choice(self.sampler_a)

            else:
                # sampler_b will be used
                out = random.choice(self.sampler_b)

            self.index += 1

        else:
            raise StopIteration

        return out


class DatasetGenerator(Dataset):
    """
    from the starter notebook
    """

    def __init__(
        self,
        metadata,
        reshape_size=64,
        label_map=[],
        dataset=[],
        transform=None,
        selected_channels=[0, 1, 2],
        dataset_image_size=None,
    ):

        self.metadata = metadata.copy().reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform
        self.selected_channels = selected_channels

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        dataset = self.metadata.loc[idx, "dataset"]
        crop_size = dataset_image_size[dataset]

        h5_file_path = self.metadata.loc[idx, "file"]
        image = imread(h5_file_path)[:, :, self.selected_channels]
        image = image / 255.0
        h1 = (image.shape[0] - crop_size) / 2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) / 2
        h2 = int(h2)

        w1 = (image.shape[1] - crop_size) / 2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) / 2
        w2 = int(w2)
        image = image[h1:h2, w1:w2, :]
        image = np.transpose(image, (2, 0, 1))
        label = self.metadata.loc[idx, "label"]

        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))
        image = image.float()

        if self.transform:
            image = self.transform(image)

        label = self.label_map[label]
        label = torch.tensor(label).long()
        return image.float(), label
