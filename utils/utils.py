import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
import torch.functional as F
from PIL import Image
from torchvision.io import read_image
import itertools
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import os
import torch.nn as nn
import umap
from torchvision.transforms.functional import pad


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def find_padding(imsize, max_w, max_h):
    h_padding = (max_w - imsize[-1]) / 2
    v_padding = (max_h - imsize[-2]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

    # padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    padding = (int(t_pad), int(l_pad), int(b_pad), int(r_pad))
    return padding


def preprocess_image(img, max_w=768, max_h=512):
    # transpose image if necessary
    img = img.rotate(90, expand=True) if img.size[0] < img.size[1] else img
    # center crop if first dim higher than 512
    # center crop if last dim is higher than 768
    w, h = img.size
    if w > max_w:
        img = T.CenterCrop((h, max_w))(img)
    if h > max_h:
        img = T.CenterCrop((max_h, w))(img)
    # pad if first dim lower than 512
    padding = find_padding(img.size, max_h, max_w)
    img = pad(img, padding, padding_mode="reflect")
    assert img.size[0] == max_w and img.size[1] == max_h
    return img


def extract_patches(batch, size=256):
    batch_size, c, h, w = batch.shape
    # default case
    stride_h = stride_w = size
    # calculate the stride base on the patch size
    assert size <= h and size <= w
    num_patches_h = math.ceil(h / size)
    num_patches_w = math.ceil(w / size)
    stride_h = math.floor((h - size) / (num_patches_h - 1))
    stride_w = math.floor((w - size) / (num_patches_w - 1))
    # print(f"stride_h: {stride_h}, stride_w: {stride_w}")

    # first in h direction
    patches = batch.unfold(-2, size, stride_h)
    # then in w direction
    patches = patches.unfold(-2, size, stride_w)
    num_patches = patches.shape[2] * patches.shape[3]
    # reshape to batch_size * num_patches * c * size * size
    patches = patches.reshape(batch_size, c, -1, size, size)
    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.reshape(-1, c, size, size)
    return patches, num_patches


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, one_hot=True):
        self.image_paths = image_paths
        if labels is None:
            self.labels = [np.nan] * len(image_paths)
        elif one_hot:
            self.labels = torch.LongTensor(np.eye(len(set(labels)))[labels])
        else:
            self.labels = labels
        self.transform = transform
        # self.images = [read_image(path) for path in image_paths]
        # self.images = [cv2.imread(path) for path in image_paths]
        self.orig_images = [Image.open(path).convert("RGB") for path in image_paths]
        self.images = [preprocess_image(img) for img in self.orig_images]
        # make sure first dimension is always shortest

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.images[index]
        if self.labels is not None:
            label = self.labels[index]
        else:
            label = np.nan

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_image_and_label_paths(image_dir, image_format="png"):
    image_paths = [
        f"{image_dir}/{fname}"
        for fname in os.listdir(image_dir)
        if fname.split(".")[-1] == image_format
    ]
    label_path = f"{image_dir}/labels.csv"
    return image_paths, label_path


def get_train_test_paths(image_paths, train_ids):
    train_set_paths = [
        path
        for path in image_paths
        for train_id in train_ids
        if path.split("_")[-1].split(".")[0] == str(train_id)
    ]
    test_set_paths = list(set(image_paths) - set(train_set_paths))
    return train_set_paths, test_set_paths


def prepare_train_test_data(
    train_set_paths,
    test_set_paths,
    labels,
    train_transforms=None,
    test_transforms=None,
    train_batch_size=64,
    test_batch_size=1024,
    num_workers=8,
    shuffle=False,
    **kwargs,
):
    # train_set_paths, test_set_paths = get_train_test_paths(image_paths, train_ids)
    train_set, train_loader = prepare_data(
        train_set_paths,
        labels=labels,
        transforms=train_transforms,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        **kwargs,
    )
    test_set, test_loader = prepare_data(
        test_set_paths,
        transforms=test_transforms,
        batch_size=test_batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        **kwargs,
    )
    return train_set, train_loader, test_set, test_loader


def prepare_data(
    image_paths,
    labels=None,
    transforms=None,
    batch_size=64,
    num_workers=8,
    shuffle=False,
    **kwargs,
):
    dataset = CustomDataset(image_paths, labels=labels, transform=transforms, **kwargs)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataset, data_loader


@torch.no_grad()
def extract_feature_sets(
    feature_extractor,
    data_loader,
    num_samples_target=None,
    patchwise=True,
    patch_size=224,
    train=True,
    device="cuda:0",
):
    feature_extractor.eval()
    if not train or num_samples_target is None:
        num_samples_target = len(data_loader.dataset)

    X = []
    y = []
    num_samples_current = 0
    for batch in itertools.cycle(data_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        if patchwise:
            patches, num_patches = extract_patches(images, size=patch_size)
            features = feature_extractor(patches)
            features = features.reshape(-1, num_patches * features.shape[-1])
        else:
            features = feature_extractor(images)
        X.append(features.detach().cpu().numpy())
        y.append(labels.detach().cpu().numpy())

        num_samples_current += X[-1].shape[0]
        if num_samples_current >= num_samples_target:
            break
    X = np.concatenate(X)
    y = np.concatenate(y)
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")
    return X, y


def prepare_data_sklearn(image_paths, labels=None, transforms=None):
    gray_scale = T.Grayscale()
    dataset = CustomDataset(image_paths, labels=labels, transform=transforms)
    X = np.concatenate(
        [gray_scale(img).numpy().flatten()[None, :] for img, label in dataset]
    )
    y = np.concatenate([np.array(label)[None, :] for img, label in dataset])
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")
    return X, y


def run_sklearn_classifier(classifier, X_train, y_train, X_test, cv=5, random_state=42):
    scorer = make_scorer(f1_score, average="micro")
    # Evaluate the pipeline using cross-validation and mean F1 score
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring=scorer)
    mean_f1 = np.mean(scores)
    print(f"F1 scores: {scores}")
    print(f"Mean F1 score: {mean_f1}")
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    train_f1 = f1_score(y_train, y_train_pred, average="micro")
    print(f"Train F1 score: {train_f1}")
    return y_train_pred, y_test_pred


def create_submission_file(
    test_set_paths, y_test_pred, submission_file_path="submission.csv"
):
    ids = [int(path.split("_")[-1].split(".")[0]) for path in test_set_paths]
    submission = pd.DataFrame({"id": ids, "malignant": y_test_pred})
    submission.to_csv(submission_file_path, index=False)

    # show label distribution
    analyze_class_distribution(submission)
    return submission


def analyze_class_distribution(labels_df):
    # bring into different form
    # expanded_df = pd.get_dummies(labels_df["malignant"])
    # # rename columns
    # print(expanded_df.columns)
    # expanded_df.columns = ["negative", "benign", "malignant"]
    # print(expanded_df.columns)
    # class_counts = expanded_df.iloc[:, 0:].sum()

    class_counts = labels_df["malignant"].value_counts()
    print(class_counts.index)
    # class_counts.index = ["negative", "benign", "malignant"]
    # define index mapping
    index_mapping = {-1: "negative", 0: "benign", 1: "malignant"}
    # map index
    class_counts.index = class_counts.index.map(index_mapping)
    print(class_counts.index)
    # Plot the bar plot
    class_counts.plot(kind="bar")
    # rotate x-labels
    plt.xticks(rotation=45)

    # Set labels and title
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.title("Class ID Counts")

    # Show the plot
    plt.show()


class FeatureExtractor:
    def __init__(self, model):
        self.model = model

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)

    def __call__(self, x):
        result = self.model(x)
        return result["out"]


def visualize_umap(features, labels):
    # Apply UMAP dimensionality reduction
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(features)

    # Create a scatter plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="viridis")
    plt.colorbar()

    # Set labels and title
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("UMAP Visualization")

    # Show the plot
    plt.show()


def check_data(data_dir="./data"):
    transform = T.Compose([T.ToTensor()])
    # transform = T.Compose([T2.ToImageTensor()])

    image_dir = f"{data_dir}/oxml-carinoma-classification"
    image_paths, label_path = get_image_and_label_paths(image_dir)
    image_paths[0:1], len(image_paths)

    labels_df = pd.read_csv(label_path)
    labels = labels_df["malignant"].tolist()
    train_ids = labels_df["id"].tolist()
    train_set_paths, test_set_paths = get_train_test_paths(
        image_paths, train_ids=train_ids
    )
    len(train_set_paths), len(test_set_paths)

    train_set, train_loader, test_set, test_loader = prepare_train_test_data(
        train_set_paths,
        test_set_paths,
        labels,
        train_transforms=transform,
        test_transforms=transform,
        train_batch_size=64,
        test_batch_size=1024,
        num_workers=8,
        one_hot=False,
    )
    return train_set, train_loader, test_set, test_loader


class ClassificationHead(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_classes=3, dropout_prob=0.0):
        super().__init__()
        self.fc_1 = nn.Linear(in_features, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, num_classes)
        self.dropout_1 = nn.Dropout(dropout_prob)
        self.dropout_2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.dropout_1(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.fc_2(x)
        return x


def compare_state_dicts(state_dict_1, state_dict_2):
    for key in state_dict_1.keys():
        if key in state_dict_2.keys():
            if not state_dict_1[key].shape == state_dict_2[key].shape:
                print(key)
                print(state_dict_1[key].shape, state_dict_2[key].shape)
        else:
            print(key)


def get_num_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
