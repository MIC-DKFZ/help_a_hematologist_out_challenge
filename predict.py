import torch
import numpy as np
import os
import glob
from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset
from imageio import imread
import numpy as np
import os
import glob
import json
from PIL import Image
from torch.utils.data import DataLoader

from models.pretrained_resnet import get_IN_resnet
from models.efficientnet import EfficientNetB0, get_EfficientNetv2m
from models.dynamic_resnet import get_resnet
from augmentation.policies.hematology import *
from base_model import seed_worker
import pandas as pd


class HematologyDataset_Testset(Dataset):
    def __init__(self, data_dir, transform=None, dset="val"):
        """
        data_dir: Path to parent_dir where the 3 dataset folders are located
        dset: "acevedo" (train on acevedo, val on matek), "matek" (train on matek, val on acevedo), "combined" (20/80 split on full data)
        train: True if train, False if validation
        """
        self.transform = transform

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

        self.mean_domains = [
            [0.8691377, 0.7415468, 0.71605015],
            [0.8209144, 0.72821516, 0.8363857],
            [0.740351, 0.6517948, 0.7791037],
        ]
        self.std_domains = [
            [0.16251542, 0.18978004, 0.07850721],
            [0.16301796, 0.25068003, 0.09190886],
            [0.18629327, 0.24896133, 0.16334666],
        ]

        self.files = glob.glob(os.path.join(data_dir, "DATA-VAL/*" if dset == "val" else "DATA-TEST/*"))

        # self.files = glob.glob(os.path.join(data_dir, "*"))  # val on train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # img = (torch.from_numpy(np.array(imread(self.files[idx])[:, :, :3])) / 255.0).permute(2, 0, 1)

        image = imread(self.files[idx])[:, :, :3]

        img = Image.fromarray(image)

        if self.transform:
            img = self.transform(img)

        return img, self.files[idx].split("/")[-1].replace(".png", ".TIF")


def load_IN_resnet(cp):

    params = {"num_classes": 11, "inference": True}

    model = get_IN_resnet(params=params, depth="18", pretrained=None)
    pretrained_dict = torch.load(cp, map_location={"cuda:0": "cpu"})

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict["state_dict"]
    pretrained_dict = {
        k.replace(".model", "").replace("module.", "").replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }

    model.load_state_dict(pretrained_dict)

    return model


def load_effnetb0(cp):

    params = {"num_classes": 11, "inference": True}

    model = EfficientNetB0(num_classes=11, hypparams=params)
    pretrained_dict = torch.load(cp, map_location={"cuda:0": "cpu"})

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict["state_dict"]
    pretrained_dict = {
        k.replace(".model", "").replace("module.", "").replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }

    model.load_state_dict(pretrained_dict)

    return model


def load_effnetV2m(cp):

    model = get_EfficientNetv2m(num_classes=11, pretrained=False)
    pretrained_dict = torch.load(cp, map_location={"cuda:0": "cpu"})

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict["state_dict"]
    # print(pretrained_dict.keys())
    # print("##################################################")
    # print(model.state_dict().keys())
    # quit()
    pretrained_dict = {
        k.replace("model.", "").replace("module.", "").replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }

    model.load_state_dict(pretrained_dict)

    return model


def load_resnet(cp):

    params = {
        "num_classes": 11,
        "inference": True,
        "cifar_size": False,
        "bottleneck": False,
        "name": "ResNet34",
        "input_channels": 3,
        "input_dim": 2,
        "resnet_dropout": 0.0,
        "stochastic_depth": 0.0,
        "squeeze_excitation": False,
        "se_rd_ratio": 0.25,
    }

    model = get_resnet(params)
    pretrained_dict = torch.load(cp, map_location={"cuda:0": "cpu"})

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict["state_dict"]
    pretrained_dict = {
        k.replace(".model", "").replace("module.", "").replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }

    model.load_state_dict(pretrained_dict)

    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--cp_dir",
        type=str,
        help="path to checkpoints",
        default="/home/s522r/Desktop/cluster_results/HematologyData/test",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="path to data",
        default="/home/s522r/Desktop/hida_challenge/Datasets",
    )
    parser.add_argument("--save_dir", default="/home/s522r/Desktop/cluster_results/HematologyData/preds")
    parser.add_argument("--set", default="val")

    args = parser.parse_args()
    cp_dir = args.cp_dir
    save_dir = args.save_dir
    dset = args.set

    checkpoints = glob.glob(os.path.join(cp_dir, "*.ckpt"))

    data = HematologyDataset_Testset(
        os.path.join(args.data_dir, "WBC1" if dset == "val" else "WBC2"), get_bg_val_transform(), dset=dset
    )
    testloader = DataLoader(
        data,
        batch_size=512,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=True,
    )

    df = pd.DataFrame(
        columns=[
            "Image",
            "softmax_0",
            "softmax_1",
            "softmax_2",
            "softmax_3",
            "softmax_4",
            "softmax_5",
            "softmax_6",
            "softmax_7",
            "softmax_8",
            "softmax_9",
            "softmax_10",
        ]
    )

    for cp in checkpoints:

        if "IN_RN" in cp:
            model = load_IN_resnet(cp)

        elif "RN34" in cp:
            model = load_resnet(cp)

        elif "efnetB0" in cp:
            model = load_effnetb0(cp)

        elif "efnetV2m" in cp:
            model = load_effnetV2m(cp)

        model = model.to("cuda")
        model.eval()

        with torch.no_grad():
            for img, filename in testloader:

                img = img.to("cuda")
                # print(img.mean())
                # print(img[0, :, 100, 100])

                output = model(img)
                output += model(torch.flip(img, (2,)))
                output += model(torch.flip(img, (3,)))
                output += model(torch.flip(img, (2, 3)))
                output /= 4
                # print(output.shape)
                output = torch.softmax(output, dim=1)
                # print(output.shape)
                # print(np.bincount(output.cpu().argmax(1), minlength=11))

                # model_preds.append(output, filename)

                out = output.detach().cpu()

                # print(out.shape)

                out_df = pd.DataFrame(
                    data={
                        "Image": filename,
                        "softmax_0": out[:, 0],
                        "softmax_1": out[:, 1],
                        "softmax_2": out[:, 2],
                        "softmax_3": out[:, 3],
                        "softmax_4": out[:, 4],
                        "softmax_5": out[:, 5],
                        "softmax_6": out[:, 6],
                        "softmax_7": out[:, 7],
                        "softmax_8": out[:, 8],
                        "softmax_9": out[:, 9],
                        "softmax_10": out[:, 10],
                    },
                )

                df = pd.concat([df, out_df])

        # print(df.shape)

    # merge rows with mean to aggregate softmax
    df = df.groupby("Image").agg("mean").reset_index()

    # print(df)
    # argmax to obtain pred
    # softmaxes = df[df.columns.difference(["Image"])].to_numpy()
    df["LabelID"] = df[df.columns.difference(["Image"])].idxmax(axis="columns").replace({"softmax_": ""}, regex=True)
    df["LabelID"] = pd.to_numeric(df["LabelID"])

    """prediction = softmaxes.argmax(1)
    print(np.bincount(prediction, minlength=11))
    df["LabelID"] = prediction
    print(df["LabelID"].value_counts())"""
    # print(df)

    label_map_reverse = {
        0: "basophil",
        1: "eosinophil",
        2: "erythroblast",
        3: "myeloblast",
        4: "promyelocyte",
        5: "myelocyte",
        6: "metamyelocyte",
        7: "neutrophil_banded",
        8: "neutrophil_segmented",
        9: "monocyte",
        10: "lymphocyte_typical",
    }

    df["Label"] = df["LabelID"].replace(label_map_reverse, regex=True)

    df = df[["Image", "Label", "LabelID"]]
    print(df["LabelID"].value_counts())
    print(df["Label"].value_counts())

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "RN34_{}_submission.csv".format(dset)))

    print("Done")
