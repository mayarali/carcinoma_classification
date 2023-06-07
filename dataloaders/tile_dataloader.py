import csv

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import einops
import os
from sklearn.model_selection import train_test_split
import torchvision
from sklearn.model_selection import KFold
from torchvision.io import read_image
from torchvision.transforms.functional import pad
def _init_fn(worker_id):
    """
    This function is fed into the dataloaders to have deterministic shuffle.
    :param worker_id:
    :return:
    """
    np.random.seed(15 + worker_id)

class OxML_Tiles_Supervised_Dataset(Dataset):
    def __init__(self, config, set_name, this_transforms=None):
        self.config = config
        self.data_split = set_name # train, test, val

        #Read png filenames on a list
        with open(self.config.dataset.data_root, "r") as file:
            self.data_filenames = file.readlines()

        #Read labels on a dict
        with open(self.config.dataset.label_root, "r") as file:
            reader = csv.reader(file)
            self.labels = {int(row[0]): int(row[1]) for row in reader if row[0] != "id"}

        #Keep only the images for which we have labels
        self.data_filenames = [i[:-1] for i in self.data_filenames if int(i.split(".")[0].split("_")[-1]) in self.labels.keys()] #-1 is to remove \n from the readlines

        #Put the labels on an array aligned with the filenames
        self.labels = np.array([self.labels[int(i.split(".")[0].split("_")[-1])] for i in self.data_filenames])+1

        self.this_transforms = this_transforms
        self._split_train_val(set=self.data_split)

        self.print_unique()

        if len(self.data_filenames)>0:
            self.images = torch.cat([self._pad_image(read_image(f)).unsqueeze(dim=0) for f in self.data_filenames], dim=0)
        else:
            self.images = torch.Tensor([])

        self.create_tiles()

    def print_unique(self):

        message = ""
        l, c = np.unique(self.labels, return_counts=True)
        for i in range(len(l)):message += "{}:{} - ".format(l[i],c[i])
        print("Split {} has {}".format(self.data_split, message[:-2]))


    def create_tiles(self):
        if len(self.data_filenames) > 0:

            #Split into patches of 28x28
            patches = self.images.unfold(3, 56, 56).unfold(2, 56, 56)
            patches = patches.contiguous().view(-1, 3, 56, 56).float()
            new_patches = patches[patches.flatten(start_dim=1).mean(dim=1) <255] #Keep only the non-zero

            new_labels = self.labels.repeat(16 * 16)[patches.flatten(start_dim=1).mean(dim=1) < 255]
            self.images = new_patches
            self.labels = new_labels
            print(self.images.shape)
            print(self.labels.shape)
            self.print_unique()

    def _split_train_val(self, set):

        if self.config.dataset.data_split.split_method == "random_stratified":
            if self.config.dataset.data_split.test_split_rate != 0 :
                X_train, X_test, y_train, y_test = train_test_split( np.array(self.data_filenames),
                                                                     self.labels,
                                                                     test_size =self.config.dataset.data_split.test_split_rate,
                                                                     random_state = self.config.training_params.seed, stratify=self.labels)
            else:
                X_train, y_train = np.array(self.data_filenames), self.labels,
                X_test, y_test = np.array([]), self.labels[0:0]

            X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                                y_train,
                                                                test_size=self.config.dataset.data_split.val_split_rate,
                                                                random_state=self.config.training_params.seed,
                                                                stratify=y_train)


            if set == "test":
                self.data_filenames = X_test
                self.labels = y_test
            elif set == "val":
                self.data_filenames = X_val
                self.labels = y_val
            elif set == "train":
                self.data_filenames = X_train
                self.labels = y_train

        elif self.config.dataset.data_split.split_method == "kfold":
            foldsplits = list(KFold(n_splits=self.config.dataset.data_split.split_fold_num, shuffle=True, random_state=self.config.training_params.seed).split(self.data_filenames))[self.config.dataset.data_split.split_fold]

            if set == "test":
                self.data_filenames = self.data_filenames[foldsplits[0]]
                self.labels = self.labels[foldsplits[0]]

            elif set == "val":
                X_train, X_val, y_train, y_val = train_test_split(self.data_filenames[foldsplits[1]],
                                                                  self.labels[foldsplits[1]],
                                                                  test_size=self.config.dataset.data_split.val_split_rate,
                                                                  random_state=self.config.training_params.seed,
                                                                  stratify=self.labels[foldsplits[1]])
                self.data_filenames = X_val
                self.labels = y_val

            elif set=="train":
                X_train, X_val, y_train, y_val = train_test_split(self.data_filenames[foldsplits[1]],
                                                                  self.labels[foldsplits[1]],
                                                                  test_size=self.config.dataset.data_split.val_split_rate,
                                                                  random_state=self.config.training_params.seed,
                                                                  stratify=self.labels[foldsplits[1]])
                self.data_filenames = X_train
                self.labels = y_train
        else:
            raise ValueError("There is no such validation split method.")

    def _pad_image(self, image):
        max_w = 896
        max_h = 896

        imsize = image.size()
        h_padding = (max_w - imsize[2]) / 2
        v_padding = (max_h - imsize[1]) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5

        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        padded_im = pad(image, padding=padding, padding_mode="constant", fill=255) # torchvision.transforms.functional.pad

        return padded_im

    def __getitem__(self, index):

        # img_f = self.data_filenames[index]
        # img = read_image(img_f)

        img = self.images[index]
        img = self.this_transforms(img)

        label = self.labels[index]

        return {"data": img, "label": label}

    def __len__(self):
        return len(self.images)

class OxML_Tiles_Unlabelled_Dataset(Dataset):
    def __init__(self, config, set_name, this_transforms=None):
        self.config = config
        self.data_split = set_name # train, test, val

        #Read png filenames on a list
        with open(self.config.dataset.data_root, "r") as file:
            self.data_filenames = file.readlines()

        #Read labels on a dict
        with open(self.config.dataset.label_root, "r") as file:
            reader = csv.reader(file)
            self.labels = {int(row[0]): int(row[1]) for row in reader if row[0] != "id"}

        #Keep only the images for which we dont have labels
        self.data_filenames = {int(i.split(".")[0].split("_")[-1]): i[:-1] for i in self.data_filenames if int(i.split(".")[0].split("_")[-1]) not in self.labels.keys()} #-1 is to remove \n from the readlines

        self.this_transforms = this_transforms

        self.images = torch.cat([self._pad_image(read_image(self.data_filenames[f])).unsqueeze(dim=0) for f in self.data_filenames], dim=0)


    def _pad_image(self, image):
        max_w = 896
        max_h = 896

        imsize = image.size()
        h_padding = (max_w - imsize[2]) / 2
        v_padding = (max_h - imsize[1]) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5

        padding = [int(l_pad), int(t_pad), int(r_pad), int(b_pad)]
        padded_im = pad(image, padding=padding) # torchvision.transforms.functional.pad

        return padded_im

    def __getitem__(self, index):

        # img_f = self.data_filenames[index]
        # img = read_image(img_f)

        img = self.images[index]
        img = self.this_transforms(img)

        id = list(self.data_filenames.keys())[index]

        return {"data": img, "id": id}

    def __len__(self):
        return len(self.images)

class OxML_Tiles_Supervised_Dataloader():

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        dataset_train, dataset_val, dataset_test, dataset_test_unlabelled, dataset_total = self._get_datasets()

        self.train_loader = torch.utils.data.DataLoader(dataset_train,
                                                        batch_size=self.config.training_params.batch_size,
                                                        num_workers=self.config.training_params.data_loader_workers,
                                                        pin_memory=self.config.training_params.pin_memory,
                                                        worker_init_fn=_init_fn)
        self.valid_loader = torch.utils.data.DataLoader(dataset_val,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=self.config.training_params.data_loader_workers,
                                                        pin_memory=self.config.training_params.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=self.config.training_params.data_loader_workers,
                                                       pin_memory=self.config.training_params.pin_memory)

        self.test_loader_unlabelled = torch.utils.data.DataLoader(dataset_test_unlabelled,
                                                       batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=self.config.training_params.data_loader_workers,
                                                       pin_memory=self.config.training_params.pin_memory)

        self.total_loader = torch.utils.data.DataLoader(dataset_total,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                        num_workers=self.config.training_params.data_loader_workers,
                                                       pin_memory=self.config.training_params.pin_memory)

    def _get_transformations(self):

        Transf_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.2073, 0.1797, 0.2261),
                                 (0.3637, 0.3280, 0.3840)),
        ])

        Transf_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.2073, 0.1797, 0.2261),
                                 (0.3637, 0.3280, 0.3840)),
            # transforms.Grayscale(num_output_channels=1),
            # transforms.Normalize(mean=[0.131015337], std=[0.30854015]),
        ])

        return {"train": Transf_train, "val": Transf_val}

    def _get_datasets(self):

        this_transforms = self._get_transformations()
        train_dataset = OxML_Tiles_Supervised_Dataset(config=self.config, set_name="train", this_transforms=this_transforms["train"])
        valid_dataset = OxML_Tiles_Supervised_Dataset(config=self.config, set_name="val", this_transforms=this_transforms["val"])
        test_dataset = OxML_Tiles_Supervised_Dataset(config=self.config, set_name="test", this_transforms=this_transforms["val"])
        test_unlabelled_dataset = OxML_Tiles_Unlabelled_Dataset(config=self.config, set_name="test", this_transforms=this_transforms["val"])
        total_dataset = OxML_Tiles_Supervised_Dataset(config=self.config, set_name="total")

        return train_dataset, valid_dataset, test_dataset, test_unlabelled_dataset, total_dataset